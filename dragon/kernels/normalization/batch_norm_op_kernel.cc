#include "dragon/core/memory.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, typename AccT, StorageOrder kOrder>
void _BatchNormExpectation(
    const std::array<int, 3>& dims,
    const AccT normalizer,
    const T* x,
    AccT* ex,
    AccT* ex2) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int count = dims[0] * dims[1] * dims[2];
  std::array<int, 3> idx = {0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const T x_val = x[i];
    const int pi = idx[kCDim];
    ex[pi] += x_val;
    ex2[pi] += x_val * x_val;
    math::utils::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
  for (int i = 0; i < dims[kCDim]; ++i) {
    ex[i] = ex[i] / normalizer;
    ex2[i] = ex2[i] / normalizer;
  }
}

template <typename T>
void _BatchNormFusedParams(
    const int C,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  EigenVectorArrayMap<T> scale_arr(scale, C);
  scale_arr = ConstEigenVectorArrayMap<T>(gamma, C) *
      ConstEigenVectorArrayMap<T>(rsig, C);
  EigenVectorArrayMap<T>(bias, C) = ConstEigenVectorArrayMap<T>(beta, C) -
      scale_arr * ConstEigenVectorArrayMap<T>(mu, C);
}

template <typename T, typename AccT>
void _BatchNormAffineNCHW(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      EigenVectorArrayMap<T>(y, S) =
          ConstEigenVectorArrayMap<T>(x, S) * scale[j] + bias[j];
      x += S;
      y += S;
    }
  }
}

template <typename T, typename AccT>
void _BatchNormAffineNHWC(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  const auto NS = N * S;
  ConstEigenVectorArrayMap<AccT> scale_arr(scale, C);
  ConstEigenVectorArrayMap<AccT> bias_arr(bias, C);
  EigenArrayMap<T>(y, C, NS) =
      (ConstEigenArrayMap<T>(x, C, NS).colwise() * scale_arr).colwise() +
      bias_arr;
}

template <typename T, typename AccT, StorageOrder kOrder>
void _BatchNormWGrad(
    const std::array<int, 3>& dims,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int count = dims[0] * dims[1] * dims[2];
  std::array<int, 3> idx = {0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const int pi = idx[kCDim];
    dgamma[pi] += dy[i] * (x[i] - mu[pi]) * rsig[pi];
    dbeta[pi] += dy[i];
    math::utils::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
void _BatchNormTrainingGrad(
    const std::array<int, 3>& dims,
    const AccT normalizer,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* dgamma,
    const AccT* dbeta,
    const T* dy,
    T* dx) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int count = dims[0] * dims[1] * dims[2];
  std::array<int, 3> idx = {0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const int pi = idx[kCDim];
    const AccT x_norm = (x[i] - mu[pi]) * rsig[pi];
    dx[i] = gamma[pi] * rsig[pi] *
        (dy[i] - (x_norm * dgamma[pi] + dbeta[pi]) / normalizer);
    math::utils::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
void _BatchNormInferenceGrad(
    const int N,
    const int C,
    const int S,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    T* dx) {
  if (kOrder == StorageOrder::NCHW) {
    const int CS = C * S;
    for (int i = 0; i < N; ++i) {
      EigenArrayMap<T>(dx + i * CS, S, C) =
          (ConstEigenArrayMap<T>(dy + i * CS, S, C).rowwise() *
           (ConstEigenVectorArrayMap<AccT>(gamma, C) *
            ConstEigenVectorArrayMap<AccT>(rsig, C))
               .transpose());
    }
  } else if (kOrder == StorageOrder::NHWC) {
    EigenArrayMap<T>(dx, C, N * S) =
        (ConstEigenArrayMap<T>(dy, C, N * S).colwise() *
         (ConstEigenVectorArrayMap<AccT>(gamma, C) *
          ConstEigenVectorArrayMap<AccT>(rsig, C)));
  }
}

} //  namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void BatchNormExpectation<float16, float, CPUContext>(
    const int N,
    const int C,
    const int S,
    const float denorm,
    const string& data_format,
    const float16* x,
    float* ex,
    float* ex2,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void BatchNorm<float16, float, CPUContext>(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const float16* x,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float* beta,
    float* scale,
    float* bias,
    float16* y,
    CPUContext* tx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void BatchNormWGrad<float16, float, CPUContext>(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const float16* x,
    const float* mu,
    const float* rsig,
    const float16* dy,
    float* dgamma,
    float* dbeta,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // BatchNormWGrad

template <>
void BatchNormTrainingGrad<float16, float, CPUContext>(
    const int N,
    const int C,
    const int S,
    const float normalizer,
    const string& data_format,
    const float16* x,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float* dgamma,
    const float* dbeta,
    const float16* dy,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // BatchNormTrainingGrad

template <>
void BatchNormInferenceGrad<float16, float, CPUContext>(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const float16* x,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float16* dy,
    float* dgamma,
    float* dbeta,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // BatchNormInferenceGrad

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                           \
  template <>                                                     \
  void BatchNorm<T, AccT, CPUContext>(                            \
      const int N,                                                \
      const int C,                                                \
      const int S,                                                \
      const string& data_format,                                  \
      const T* x,                                                 \
      const AccT* mu,                                             \
      const AccT* rsig,                                           \
      const AccT* gamma,                                          \
      const AccT* beta,                                           \
      AccT* scale,                                                \
      AccT* bias,                                                 \
      T* y,                                                       \
      CPUContext* ctx) {                                          \
    _BatchNormFusedParams(C, mu, rsig, gamma, beta, scale, bias); \
    if (data_format == "NCHW") {                                  \
      _BatchNormAffineNCHW(N, C, S, x, scale, bias, y);           \
    } else if (data_format == "NHWC") {                           \
      _BatchNormAffineNHWC(N, C, S, x, scale, bias, y);           \
    }                                                             \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                                 \
  template <>                                                                \
  void BatchNormExpectation<T, AccT, CPUContext>(                            \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const float normalizer,                                                \
      const string& data_format,                                             \
      const T* x,                                                            \
      AccT* ex,                                                              \
      AccT* ex2,                                                             \
      CPUContext* ctx) {                                                     \
    math::Set(C, AccT(0), ex, ctx);                                          \
    math::Set(C, AccT(0), ex2, ctx);                                         \
    if (data_format == "NCHW") {                                             \
      _BatchNormExpectation<T, AccT, StorageOrder::NCHW>(                    \
          {N, C, S}, normalizer, x, ex, ex2);                                \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormExpectation<T, AccT, StorageOrder::NHWC>(                    \
          {N, S, C}, normalizer, x, ex, ex2);                                \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void BatchNormWGrad<T, AccT, CPUContext>(                                  \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const T* x,                                                            \
      const AccT* mu,                                                        \
      const AccT* rsig,                                                      \
      const T* dy,                                                           \
      AccT* dgamma,                                                          \
      AccT* dbeta,                                                           \
      CPUContext* ctx) {                                                     \
    math::Set(C, AccT(0), dgamma, ctx);                                      \
    math::Set(C, AccT(0), dbeta, ctx);                                       \
    if (data_format == "NCHW") {                                             \
      _BatchNormWGrad<T, AccT, StorageOrder::NCHW>(                          \
          {N, C, S}, x, mu, rsig, dy, dgamma, dbeta);                        \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormWGrad<T, AccT, StorageOrder::NHWC>(                          \
          {N, S, C}, x, mu, rsig, dy, dgamma, dbeta);                        \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void BatchNormTrainingGrad<T, AccT, CPUContext>(                           \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const float normalizer,                                                \
      const string& data_format,                                             \
      const T* x,                                                            \
      const AccT* mu,                                                        \
      const AccT* rsig,                                                      \
      const AccT* gamma,                                                     \
      const AccT* dgamma,                                                    \
      const AccT* dbeta,                                                     \
      const T* dy,                                                           \
      T* dx,                                                                 \
      CPUContext* ctx) {                                                     \
    if (data_format == "NCHW") {                                             \
      _BatchNormTrainingGrad<T, AccT, StorageOrder::NCHW>(                   \
          {N, C, S}, normalizer, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormTrainingGrad<T, AccT, StorageOrder::NHWC>(                   \
          {N, S, C}, normalizer, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void BatchNormInferenceGrad<T, AccT, CPUContext>(                          \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const T* x,                                                            \
      const AccT* mu,                                                        \
      const AccT* rsig,                                                      \
      const AccT* gamma,                                                     \
      const T* dy,                                                           \
      AccT* dgamma,                                                          \
      AccT* dbeta,                                                           \
      T* dx,                                                                 \
      CPUContext* ctx) {                                                     \
    if (data_format == "NCHW") {                                             \
      if (dgamma != nullptr) {                                               \
        math::Set(C, AccT(0), dgamma, ctx);                                  \
        math::Set(C, AccT(0), dbeta, ctx);                                   \
        _BatchNormWGrad<T, AccT, StorageOrder::NCHW>(                        \
            {N, C, S}, x, mu, rsig, dy, dgamma, dbeta);                      \
      }                                                                      \
      _BatchNormInferenceGrad<T, AccT, StorageOrder::NCHW>(                  \
          N, C, S, rsig, gamma, dy, dx);                                     \
    } else if (data_format == "NHWC") {                                      \
      if (dgamma != nullptr) {                                               \
        math::Set(C, AccT(0), dgamma, ctx);                                  \
        math::Set(C, AccT(0), dbeta, ctx);                                   \
        _BatchNormWGrad<T, AccT, StorageOrder::NHWC>(                        \
            {N, S, C}, x, mu, rsig, dy, dgamma, dbeta);                      \
      }                                                                      \
      _BatchNormInferenceGrad<T, AccT, StorageOrder::NHWC>(                  \
          N, C, S, rsig, gamma, dy, dx);                                     \
    }                                                                        \
  }

DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
