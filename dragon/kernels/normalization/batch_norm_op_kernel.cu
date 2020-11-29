#ifdef USE_CUDA

#include "dragon/core/memory.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) convert::To<AccT>(__ldg(x + i))
#else
#define LDG(x, i) x[i]
#define LDG2(x, i) convert::To<AccT>(x[i])
#endif

namespace {

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormExpectation(
    const int N,
    const int C,
    const int S,
    const AccT denorm,
    const T* x,
    AccT* ex,
    AccT* ex2) {
  const int outer_dim = N * S;
  __shared__ union {
    typename BlockReduce<AccT>::TempStorage ex;
    typename BlockReduce<AccT>::TempStorage ex2;
  } storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    AccT ex_val = AccT(0), ex2_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
      const int xi = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                  : j * C + i;
      ex_val += LDG2(x, xi);
      ex2_val += math::utils::Square(LDG2(x, xi));
    }
    ex_val = BlockReduce<AccT>(storage.ex).Reduce(ex_val, cub::Sum());
    ex2_val = BlockReduce<AccT>(storage.ex2).Reduce(ex2_val, cub::Sum());
    if (threadIdx.x == 0) {
      ex[i] = ex_val * denorm;
      ex2[i] = ex2_val * denorm;
    }
  }
}

template <typename T>
__global__ void _BatchNormFusedParams(
    const int C,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  CUDA_1D_KERNEL_LOOP(i, C) {
    const T scale_val = scale[i] = gamma[i] * rsig[i];
    bias[i] = fma(-scale_val, mu[i], beta[i]);
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormAffine(
    const int nthreads,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int pi = kOrder == StorageOrder::NCHW ? (i / S) % C : i % C;
    y[i] = convert::To<T>(
        fma(convert::To<AccT>(x[i]), LDG(scale, pi), LDG(bias, pi)));
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormInternalGrad(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta) {
  const int outer_dim = N * S;
  __shared__ union {
    typename BlockReduce<AccT>::TempStorage dg;
    typename BlockReduce<AccT>::TempStorage db;
  } storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    AccT dg_val = AccT(0), db_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
      const int xi = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                  : j * C + i;
      dg_val += LDG2(dy, xi) * (LDG2(x, xi) - LDG(mu, i)) * LDG(rsig, i);
      db_val += LDG2(dy, xi);
    }
    dg_val = BlockReduce<AccT>(storage.dg).Reduce(dg_val, cub::Sum());
    db_val = BlockReduce<AccT>(storage.db).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormTrainingGrad(
    const int nthreads,
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* dgamma,
    const AccT* dbeta,
    const T* dy,
    T* dx) {
  const AccT denom = AccT(1) / AccT(N * S);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int pi = kOrder == StorageOrder::NCHW ? (i / S) % C : i % C;
    const AccT xnorm = (LDG2(x, i) - LDG(mu, pi)) * LDG(rsig, pi);
    dx[i] = convert::To<T>(
        LDG(gamma, pi) * LDG(rsig, pi) *
        (LDG2(dy, i) - fma(xnorm, LDG(dgamma, pi), LDG(dbeta, pi)) * denom));
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormWGrad(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta) {
  const int outer_dim = N * S;
  __shared__ union {
    typename BlockReduce<AccT>::TempStorage dg;
    typename BlockReduce<AccT>::TempStorage db;
  } storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    AccT dg_val = AccT(0), db_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, outer_dim) {
      const int xi = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                  : j * C + i;
      dg_val += LDG2(dy, xi) * (LDG2(x, xi) - LDG(mu, i)) * LDG(rsig, i);
      db_val += LDG2(dy, xi);
    }
    dg_val = BlockReduce<AccT>(storage.db).Reduce(dg_val, cub::Sum());
    db_val = BlockReduce<AccT>(storage.db).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormInferenceGrad(
    const int nthreads,
    const int C,
    const int S,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int pi = kOrder == StorageOrder::NCHW ? (i / S) % C : i % C;
    dx[i] = convert::To<T>(LDG(gamma, pi) * LDG2(dy, i) * LDG(rsig, pi));
  }
}

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_BATCHNORM_KERNEL(name, T, AccT, nblocks, nthreads, ...) \
  if (data_format == "NCHW") {                                           \
    name<T, AccT, StorageOrder::NCHW>                                    \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else if (data_format == "NHWC") {                                    \
    name<T, AccT, StorageOrder::NHWC>                                    \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else {                                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                 \
  }

#define DEFINE_KERNEL_LAUNCHER(T, ScalarT, AccT)                      \
  template <>                                                         \
  void BatchNorm<T, AccT, CUDAContext>(                               \
      const int N,                                                    \
      const int C,                                                    \
      const int S,                                                    \
      const string& data_format,                                      \
      const T* x,                                                     \
      const AccT* mu,                                                 \
      const AccT* rsig,                                               \
      const AccT* gamma,                                              \
      const AccT* beta,                                               \
      AccT* scale,                                                    \
      AccT* bias,                                                     \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    const auto nthreads = N * C * S;                                  \
    _BatchNormFusedParams<<<                                          \
        CUDA_BLOCKS(C),                                               \
        CUDA_THREADS,                                                 \
        0,                                                            \
        ctx->cuda_stream()>>>(C, mu, rsig, gamma, beta, scale, bias); \
    DISPATCH_BATCHNORM_KERNEL(                                        \
        _BatchNormAffine,                                             \
        ScalarT,                                                      \
        AccT,                                                         \
        CUDA_BLOCKS(nthreads),                                        \
        CUDA_THREADS,                                                 \
        nthreads,                                                     \
        C,                                                            \
        S,                                                            \
        reinterpret_cast<const ScalarT*>(x),                          \
        scale,                                                        \
        bias,                                                         \
        reinterpret_cast<ScalarT*>(y));                               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, ScalarT, AccT) \
  template <>                                         \
  void BatchNormExpectation<T, AccT, CUDAContext>(    \
      const int N,                                    \
      const int C,                                    \
      const int S,                                    \
      const AccT denorm,                              \
      const string& data_format,                      \
      const T* x,                                     \
      AccT* ex,                                       \
      AccT* ex2,                                      \
      CUDAContext* ctx) {                             \
    DISPATCH_BATCHNORM_KERNEL(                        \
        _BatchNormExpectation,                        \
        ScalarT,                                      \
        AccT,                                         \
        CUDA_2D_BLOCKS(C),                            \
        CUDA_THREADS,                                 \
        N,                                            \
        C,                                            \
        S,                                            \
        denorm,                                       \
        reinterpret_cast<const ScalarT*>(x),          \
        ex,                                           \
        ex2);                                         \
  }                                                   \
  template <>                                         \
  void BatchNormInternalGrad<T, AccT, CUDAContext>(   \
      const int N,                                    \
      const int C,                                    \
      const int S,                                    \
      const string& data_format,                      \
      const T* x,                                     \
      const AccT* mu,                                 \
      const AccT* rsig,                               \
      const AccT* gamma,                              \
      const T* dy,                                    \
      AccT* dgamma,                                   \
      AccT* dbeta,                                    \
      CUDAContext* ctx) {                             \
    DISPATCH_BATCHNORM_KERNEL(                        \
        _BatchNormInternalGrad,                       \
        ScalarT,                                      \
        AccT,                                         \
        CUDA_2D_BLOCKS(C),                            \
        CUDA_THREADS,                                 \
        N,                                            \
        C,                                            \
        S,                                            \
        reinterpret_cast<const ScalarT*>(x),          \
        mu,                                           \
        rsig,                                         \
        gamma,                                        \
        reinterpret_cast<const ScalarT*>(dy),         \
        dgamma,                                       \
        dbeta);                                       \
  }                                                   \
  template <>                                         \
  void BatchNormTrainingGrad<T, AccT, CUDAContext>(   \
      const int N,                                    \
      const int C,                                    \
      const int S,                                    \
      const string& data_format,                      \
      const T* x,                                     \
      const AccT* mu,                                 \
      const AccT* rsig,                               \
      const AccT* gamma,                              \
      const AccT* dgamma,                             \
      const AccT* dbeta,                              \
      const T* dy,                                    \
      T* dx,                                          \
      CUDAContext* ctx) {                             \
    const auto nthreads = N * C * S;                  \
    DISPATCH_BATCHNORM_KERNEL(                        \
        _BatchNormTrainingGrad,                       \
        ScalarT,                                      \
        AccT,                                         \
        CUDA_BLOCKS(nthreads),                        \
        CUDA_THREADS,                                 \
        nthreads,                                     \
        N,                                            \
        C,                                            \
        S,                                            \
        reinterpret_cast<const ScalarT*>(x),          \
        mu,                                           \
        rsig,                                         \
        gamma,                                        \
        dgamma,                                       \
        dbeta,                                        \
        reinterpret_cast<const ScalarT*>(dy),         \
        reinterpret_cast<ScalarT*>(dx));              \
  }                                                   \
  template <>                                         \
  void BatchNormInferenceGrad<T, AccT, CUDAContext>(  \
      const int N,                                    \
      const int C,                                    \
      const int S,                                    \
      const string& data_format,                      \
      const T* x,                                     \
      const AccT* mu,                                 \
      const AccT* rsig,                               \
      const AccT* gamma,                              \
      const T* dy,                                    \
      AccT* dgamma,                                   \
      AccT* dbeta,                                    \
      T* dx,                                          \
      CUDAContext* ctx) {                             \
    const auto nthreads = N * C * S;                  \
    if (dgamma != nullptr) {                          \
      DISPATCH_BATCHNORM_KERNEL(                      \
          _BatchNormWGrad,                            \
          ScalarT,                                    \
          AccT,                                       \
          CUDA_2D_BLOCKS(C),                          \
          CUDA_THREADS,                               \
          N,                                          \
          C,                                          \
          S,                                          \
          reinterpret_cast<const ScalarT*>(x),        \
          mu,                                         \
          rsig,                                       \
          reinterpret_cast<const ScalarT*>(dy),       \
          dgamma,                                     \
          dbeta);                                     \
    }                                                 \
    DISPATCH_BATCHNORM_KERNEL(                        \
        _BatchNormInferenceGrad,                      \
        ScalarT,                                      \
        AccT,                                         \
        CUDA_BLOCKS(nthreads),                        \
        CUDA_THREADS,                                 \
        nthreads,                                     \
        C,                                            \
        S,                                            \
        rsig,                                         \
        gamma,                                        \
        reinterpret_cast<const ScalarT*>(dy),         \
        reinterpret_cast<ScalarT*>(dx));              \
  }

DEFINE_KERNEL_LAUNCHER(float16, half, float);
DEFINE_KERNEL_LAUNCHER(float, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, half, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float, float);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_BATCHNORM_KERNEL

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
