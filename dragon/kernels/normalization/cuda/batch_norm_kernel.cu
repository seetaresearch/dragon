#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

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
      ex_val += math::utils::LDGC<AccT>(x + xi);
      ex2_val += math::utils::Sqr(math::utils::LDGC<AccT>(x + xi));
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
    y[i] = fma(convert::To<AccT>(x[i]), __ldg(scale + j), __ldg(bias + j));
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
    CUDA_2D_KERNEL_LOOP2(j, NxS) { // clang-format off
      const int idx = kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                                   : j * C + i;

      dg_val += math::utils::LDGC<AccT>(dy + idx) * (
                    math::utils::LDGC<AccT>(x + idx) - __ldg(mu + i));
      db_val += math::utils::LDGC<AccT>(dy + idx);
    } // clang-format on
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
  CUDA_1D_KERNEL_LOOP(i, NxCxS) { // clang-format off
    const int j = kOrder == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] = __ldg(gamma + j) * __ldg(rsig + j) * (
        convert::To<AccT>(dy[i]) - fma(
            (convert::To<AccT>(x[i]) - __ldg(mu + j)) * __ldg(rsig + j),
            __ldg(dgamma + j), __ldg(dbeta + j)) /  normalizer);
  } // clang-format on
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
    dx[i] = __ldg(gamma + j) * convert::To<AccT>(dy[i]) * __ldg(rsig + j);
  }
}

} // namespace

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

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                           \
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
        math::Traits<T>::scalar_type,                             \
        AccT,                                                     \
        C,                                                        \
        CUDA_THREADS,                                             \
        N,                                                        \
        C,                                                        \
        S,                                                        \
        normalizer,                                               \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x), \
        ex,                                                       \
        ex2);                                                     \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

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
    if (scale == rsig) {                                              \
      scale = (AccT*)rsig + C, bias = (AccT*)rsig + C * 2;            \
    } else if (scale == bias) {                                       \
      bias = scale + C;                                               \
    }                                                                 \
    _BatchNormFusedParams<<<                                          \
        CUDA_BLOCKS(C),                                               \
        CUDA_THREADS,                                                 \
        0,                                                            \
        ctx->cuda_stream()>>>(C, mu, rsig, gamma, beta, scale, bias); \
    DISPATCH_BATCHNORM_KERNEL(                                        \
        _BatchNormAffine,                                             \
        math::Traits<T>::scalar_type,                                 \
        AccT,                                                         \
        CUDA_BLOCKS(NxCxS),                                           \
        CUDA_THREADS,                                                 \
        NxCxS,                                                        \
        C,                                                            \
        S,                                                            \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),     \
        scale,                                                        \
        bias,                                                         \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));          \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                       \
  template <>                                                      \
  void BatchNormWGrad<T, AccT, CUDAContext>(                       \
      const int N,                                                 \
      const int C,                                                 \
      const int S,                                                 \
      const string& data_format,                                   \
      const T* x,                                                  \
      const AccT* mu,                                              \
      const AccT* rsig,                                            \
      const T* dy,                                                 \
      AccT* dgamma,                                                \
      AccT* dbeta,                                                 \
      CUDAContext* ctx) {                                          \
    dbeta = (dgamma == dbeta ? dgamma + C : dbeta);                \
    DISPATCH_BATCHNORM_KERNEL(                                     \
        _BatchNormWGrad,                                           \
        math::Traits<T>::scalar_type,                              \
        AccT,                                                      \
        C,                                                         \
        CUDA_THREADS,                                              \
        N,                                                         \
        C,                                                         \
        S,                                                         \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),  \
        mu,                                                        \
        rsig,                                                      \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy), \
        dgamma,                                                    \
        dbeta);                                                    \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                       \
  template <>                                                      \
  void BatchNormTrainingGrad<T, AccT, CUDAContext>(                \
      const int N,                                                 \
      const int C,                                                 \
      const int S,                                                 \
      const float normalizer,                                      \
      const string& data_format,                                   \
      const T* x,                                                  \
      const AccT* mu,                                              \
      const AccT* rsig,                                            \
      const AccT* gamma,                                           \
      const AccT* dgamma,                                          \
      const AccT* dbeta,                                           \
      const T* dy,                                                 \
      T* dx,                                                       \
      CUDAContext* ctx) {                                          \
    const auto NxCxS = N * C * S;                                  \
    dbeta = (dgamma == dbeta ? dgamma + C : dbeta);                \
    DISPATCH_BATCHNORM_KERNEL(                                     \
        _BatchNormTrainingGrad,                                    \
        math::Traits<T>::scalar_type,                              \
        AccT,                                                      \
        CUDA_BLOCKS(NxCxS),                                        \
        CUDA_THREADS,                                              \
        NxCxS,                                                     \
        C,                                                         \
        S,                                                         \
        normalizer,                                                \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),  \
        mu,                                                        \
        rsig,                                                      \
        gamma,                                                     \
        dgamma,                                                    \
        dbeta,                                                     \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy), \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));      \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                          \
  template <>                                                         \
  void BatchNormInferenceGrad<T, AccT, CUDAContext>(                  \
      const int N,                                                    \
      const int C,                                                    \
      const int S,                                                    \
      const string& data_format,                                      \
      const T* x,                                                     \
      const AccT* mu,                                                 \
      const AccT* rsig,                                               \
      const AccT* gamma,                                              \
      const T* dy,                                                    \
      AccT* dgamma,                                                   \
      AccT* dbeta,                                                    \
      T* dx,                                                          \
      CUDAContext* ctx) {                                             \
    const auto NxCxS = N * C * S;                                     \
    if (dgamma != nullptr) {                                          \
      BatchNormWGrad<T, AccT>(                                        \
          N, C, S, data_format, x, mu, rsig, dy, dgamma, dbeta, ctx); \
    }                                                                 \
    DISPATCH_BATCHNORM_KERNEL(                                        \
        _BatchNormInferenceGrad,                                      \
        math::Traits<T>::scalar_type,                                 \
        AccT,                                                         \
        CUDA_BLOCKS(NxCxS),                                           \
        CUDA_THREADS,                                                 \
        NxCxS,                                                        \
        C,                                                            \
        S,                                                            \
        rsig,                                                         \
        gamma,                                                        \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),    \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));         \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_BATCHNORM_KERNEL

} // namespace kernels

} // namespace dragon
