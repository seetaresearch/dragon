#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) convert::To<AccT>(__ldg(x + i))

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormExpectation(
    const int N,
    const int C,
    const int S,
    const AccT normalizer,
    const T* x,
    AccT* ex,
    AccT* ex2) {
  const int NxS = N * S;
  __shared__ typename BlockReduce<AccT>::TempStorage ex_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage ex2_storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    AccT ex_val = AccT(0), ex2_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, NxS) {
      const int xi = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                  : j * C + i;
      ex_val += LDG2(x, xi);
      ex2_val += math::utils::Square(LDG2(x, xi));
    }
    ex_val = BlockReduce<AccT>(ex_storage).Reduce(ex_val, cub::Sum());
    ex2_val = BlockReduce<AccT>(ex2_storage).Reduce(ex2_val, cub::Sum());
    if (threadIdx.x == 0) {
      ex[i] = ex_val / normalizer;
      ex2[i] = ex2_val / normalizer;
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
    const T val = scale[i] = gamma[i] * rsig[i];
    bias[i] = fma(-val, mu[i], beta[i]);
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormAffine(
    const int NxCxS,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    y[i] = convert::To<T>(
        fma(convert::To<AccT>(x[i]), LDG(scale, j), LDG(bias, j)));
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
  const int NxS = N * S;
  __shared__ typename BlockReduce<AccT>::TempStorage dg_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage db_storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    AccT dg_val = AccT(0), db_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, NxS) {
      const int idx = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                   : j * C + i;
      dg_val += LDG2(dy, idx) * (LDG2(x, idx) - LDG(mu, i));
      db_val += LDG2(dy, idx);
    }
    dg_val = BlockReduce<AccT>(dg_storage).Sum(dg_val);
    db_val = BlockReduce<AccT>(db_storage).Sum(db_val);
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val * rsig[i];
      dbeta[i] = db_val;
    }
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormTrainingGrad(
    const int NxCxS,
    const int C,
    const int S,
    const AccT normalizer,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* dgamma,
    const AccT* dbeta,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] = convert::To<T>(
        LDG(gamma, j) * LDG(rsig, j) *
        (convert::To<AccT>(dy[i]) -
         fma((convert::To<AccT>(x[i]) - LDG(mu, j)) * LDG(rsig, j),
             LDG(dgamma, j),
             LDG(dbeta, j)) /
             normalizer));
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _BatchNormInferenceGrad(
    const int NxCxS,
    const int C,
    const int S,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] =
        convert::To<T>(LDG(gamma, j) * convert::To<AccT>(dy[i]) * LDG(rsig, j));
  }
}

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_BATCHNORM_KERNEL(name, T, AccT, kBlocks, kThreads, ...) \
  if (data_format == "NCHW") {                                           \
    name<T, AccT, StorageOrder::NCHW>                                    \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else if (data_format == "NHWC") {                                    \
    name<T, AccT, StorageOrder::NHWC>                                    \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);     \
  } else {                                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                 \
  }

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                               \
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
    const auto NxCxS = N * C * S;                                     \
    _BatchNormFusedParams<<<                                          \
        CUDA_BLOCKS(C),                                               \
        CUDA_THREADS,                                                 \
        0,                                                            \
        ctx->cuda_stream()>>>(C, mu, rsig, gamma, beta, scale, bias); \
    DISPATCH_BATCHNORM_KERNEL(                                        \
        _BatchNormAffine,                                             \
        math::ScalarType<T>::type,                                    \
        AccT,                                                         \
        CUDA_BLOCKS(NxCxS),                                           \
        CUDA_THREADS,                                                 \
        NxCxS,                                                        \
        C,                                                            \
        S,                                                            \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),        \
        scale,                                                        \
        bias,                                                         \
        reinterpret_cast<math::ScalarType<T>::type*>(y));             \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                      \
  template <>                                                     \
  void BatchNormExpectation<T, AccT, CUDAContext>(                \
      const int N,                                                \
      const int C,                                                \
      const int S,                                                \
      const float normalizer,                                     \
      const string& data_format,                                  \
      const T* x,                                                 \
      AccT* ex,                                                   \
      AccT* ex2,                                                  \
      CUDAContext* ctx) {                                         \
    DISPATCH_BATCHNORM_KERNEL(                                    \
        _BatchNormExpectation,                                    \
        math::ScalarType<T>::type,                                \
        AccT,                                                     \
        C,                                                        \
        CUDA_THREADS,                                             \
        N,                                                        \
        C,                                                        \
        S,                                                        \
        normalizer,                                               \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),    \
        ex,                                                       \
        ex2);                                                     \
  }                                                               \
  template <>                                                     \
  void BatchNormWGrad<T, AccT, CUDAContext>(                      \
      const int N,                                                \
      const int C,                                                \
      const int S,                                                \
      const string& data_format,                                  \
      const T* x,                                                 \
      const AccT* mu,                                             \
      const AccT* rsig,                                           \
      const T* dy,                                                \
      AccT* dgamma,                                               \
      AccT* dbeta,                                                \
      CUDAContext* ctx) {                                         \
    DISPATCH_BATCHNORM_KERNEL(                                    \
        _BatchNormWGrad,                                          \
        math::ScalarType<T>::type,                                \
        AccT,                                                     \
        C,                                                        \
        CUDA_THREADS,                                             \
        N,                                                        \
        C,                                                        \
        S,                                                        \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),    \
        mu,                                                       \
        rsig,                                                     \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),   \
        dgamma,                                                   \
        dbeta);                                                   \
  }                                                               \
  template <>                                                     \
  void BatchNormTrainingGrad<T, AccT, CUDAContext>(               \
      const int N,                                                \
      const int C,                                                \
      const int S,                                                \
      const float normalizer,                                     \
      const string& data_format,                                  \
      const T* x,                                                 \
      const AccT* mu,                                             \
      const AccT* rsig,                                           \
      const AccT* gamma,                                          \
      const AccT* dgamma,                                         \
      const AccT* dbeta,                                          \
      const T* dy,                                                \
      T* dx,                                                      \
      CUDAContext* ctx) {                                         \
    const auto NxCxS = N * C * S;                                 \
    DISPATCH_BATCHNORM_KERNEL(                                    \
        _BatchNormTrainingGrad,                                   \
        math::ScalarType<T>::type,                                \
        AccT,                                                     \
        CUDA_BLOCKS(NxCxS),                                       \
        CUDA_THREADS,                                             \
        NxCxS,                                                    \
        C,                                                        \
        S,                                                        \
        normalizer,                                               \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),    \
        mu,                                                       \
        rsig,                                                     \
        gamma,                                                    \
        dgamma,                                                   \
        dbeta,                                                    \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),   \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));        \
  }                                                               \
  template <>                                                     \
  void BatchNormInferenceGrad<T, AccT, CUDAContext>(              \
      const int N,                                                \
      const int C,                                                \
      const int S,                                                \
      const string& data_format,                                  \
      const T* x,                                                 \
      const AccT* mu,                                             \
      const AccT* rsig,                                           \
      const AccT* gamma,                                          \
      const T* dy,                                                \
      AccT* dgamma,                                               \
      AccT* dbeta,                                                \
      T* dx,                                                      \
      CUDAContext* ctx) {                                         \
    const auto NxCxS = N * C * S;                                 \
    if (dgamma != nullptr) {                                      \
      DISPATCH_BATCHNORM_KERNEL(                                  \
          _BatchNormWGrad,                                        \
          math::ScalarType<T>::type,                              \
          AccT,                                                   \
          C,                                                      \
          CUDA_THREADS,                                           \
          N,                                                      \
          C,                                                      \
          S,                                                      \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),  \
          mu,                                                     \
          rsig,                                                   \
          reinterpret_cast<const math::ScalarType<T>::type*>(dy), \
          dgamma,                                                 \
          dbeta);                                                 \
    }                                                             \
    DISPATCH_BATCHNORM_KERNEL(                                    \
        _BatchNormInferenceGrad,                                  \
        math::ScalarType<T>::type,                                \
        AccT,                                                     \
        CUDA_BLOCKS(NxCxS),                                       \
        CUDA_THREADS,                                             \
        NxCxS,                                                    \
        C,                                                        \
        S,                                                        \
        rsig,                                                     \
        gamma,                                                    \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),   \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));        \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_BATCHNORM_KERNEL

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
