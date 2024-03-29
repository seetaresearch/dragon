#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT, StorageOrder kOrder>
void _BatchNormExpectation(
    const std::array<int, 3>& dims,
    const AccT normalizer,
    const T* x,
    AccT* ex,
    AccT* ex2) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int C = dims[kCDim];
  const int NxCxS = dims[0] * dims[1] * dims[2];
  std::array<int, 3> index = {0, 0, 0};
  for (int xi = 0; xi < NxCxS; ++xi) {
    const T val = x[xi];
    const int i = index[kCDim];
    ex[i] += val;
    ex2[i] += val * val;
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
  for (int i = 0; i < C; ++i) {
    ex[i] /= normalizer;
    ex2[i] /= normalizer;
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
  EigenVectorArrayMap<T> Scale(scale, C);
  EigenVectorArrayMap<T> Bias(bias, C);
  Scale = ConstEigenVectorArrayMap<T>(gamma, C) *
      ConstEigenVectorArrayMap<T>(rsig, C);
  Bias = ConstEigenVectorArrayMap<T>(beta, C) -
      Scale * ConstEigenVectorArrayMap<T>(mu, C);
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
  ConstEigenVectorArrayMap<AccT> Scale(scale, C);
  ConstEigenVectorArrayMap<AccT> Bias(bias, C);
  ConstEigenArrayMap<T> X(x, C, NS);
  EigenArrayMap<T>(y, C, NS) = (X.colwise() * Scale).colwise() + Bias;
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
  const int NxCxS = dims[0] * dims[1] * dims[2];
  std::array<int, 3> index = {0, 0, 0};
  for (int i = 0; i < NxCxS; ++i) {
    const int pi = index[kCDim];
    dgamma[pi] += dy[i] * (x[i] - mu[pi]) * rsig[pi];
    dbeta[pi] += dy[i];
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
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
  const int NxCxS = dims[0] * dims[1] * dims[2];
  std::array<int, 3> index = {0, 0, 0};
  for (int i = 0; i < NxCxS; ++i) {
    const int pi = index[kCDim];
    const AccT x_norm = (x[i] - mu[pi]) * rsig[pi];
    dx[i] = gamma[pi] * rsig[pi] *
        (dy[i] - (x_norm * dgamma[pi] + dbeta[pi]) / normalizer);
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
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

#define DEFINE_KERNEL_LAUNCHER(T, AccT)           \
  template <>                                     \
  void BatchNormExpectation<T, AccT, CPUContext>( \
      const int N,                                \
      const int C,                                \
      const int S,                                \
      const float denorm,                         \
      const string& data_format,                  \
      const T* x,                                 \
      float* ex,                                  \
      float* ex2,                                 \
      CPUContext* ctx) {                          \
    CPU_UNSUPPORTED_DTYPE(T);                     \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                   \
  template <>                                             \
  void BatchNormExpectation<T, AccT, CPUContext>(         \
      const int N,                                        \
      const int C,                                        \
      const int S,                                        \
      const float normalizer,                             \
      const string& data_format,                          \
      const T* x,                                         \
      AccT* ex,                                           \
      AccT* ex2,                                          \
      CPUContext* ctx) {                                  \
    math::Set(C, AccT(0), ex, ctx);                       \
    math::Set(C, AccT(0), ex2, ctx);                      \
    if (data_format == "NCHW") {                          \
      _BatchNormExpectation<T, AccT, StorageOrder::NCHW>( \
          {N, C, S}, normalizer, x, ex, ex2);             \
    } else if (data_format == "NHWC") {                   \
      _BatchNormExpectation<T, AccT, StorageOrder::NHWC>( \
          {N, S, C}, normalizer, x, ex, ex2);             \
    }                                                     \
  }

DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T, AccT) \
  template <>                           \
  void BatchNorm<T, AccT, CPUContext>(  \
      const int N,                      \
      const int C,                      \
      const int S,                      \
      const string& data_format,        \
      const T* x,                       \
      const AccT* mu,                   \
      const AccT* rsig,                 \
      const AccT* gamma,                \
      const AccT* beta,                 \
      AccT* scale,                      \
      AccT* bias,                       \
      T* y,                             \
      CPUContext* ctx) {                \
    CPU_UNSUPPORTED_DTYPE(T);           \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
#undef DEFINE_KERNEL_LAUNCHER

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
    if (scale == rsig) {                                          \
      scale = (AccT*)rsig + C, bias = (AccT*)rsig + C * 2;        \
    } else if (scale == bias) {                                   \
      bias = scale + C;                                           \
    }                                                             \
    _BatchNormFusedParams(C, mu, rsig, gamma, beta, scale, bias); \
    if (data_format == "NCHW") {                                  \
      _BatchNormAffineNCHW(N, C, S, x, scale, bias, y);           \
    } else if (data_format == "NHWC") {                           \
      _BatchNormAffineNHWC(N, C, S, x, scale, bias, y);           \
    }                                                             \
  }

DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT) \
  template <>                                \
  void BatchNormWGrad<T, AccT, CPUContext>(  \
      const int N,                           \
      const int C,                           \
      const int S,                           \
      const string& data_format,             \
      const T* x,                            \
      const AccT* mu,                        \
      const AccT* rsig,                      \
      const T* dy,                           \
      AccT* dgamma,                          \
      AccT* dbeta,                           \
      CPUContext* ctx) {                     \
    CPU_UNSUPPORTED_DTYPE(T);                \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)          \
  template <>                                         \
  void BatchNormWGrad<T, AccT, CPUContext>(           \
      const int N,                                    \
      const int C,                                    \
      const int S,                                    \
      const string& data_format,                      \
      const T* x,                                     \
      const AccT* mu,                                 \
      const AccT* rsig,                               \
      const T* dy,                                    \
      AccT* dgamma,                                   \
      AccT* dbeta,                                    \
      CPUContext* ctx) {                              \
    dbeta = (dgamma == dbeta ? dgamma + C : dbeta);   \
    math::Set(C, AccT(0), dgamma, ctx);               \
    math::Set(C, AccT(0), dbeta, ctx);                \
    if (data_format == "NCHW") {                      \
      _BatchNormWGrad<T, AccT, StorageOrder::NCHW>(   \
          {N, C, S}, x, mu, rsig, dy, dgamma, dbeta); \
    } else if (data_format == "NHWC") {               \
      _BatchNormWGrad<T, AccT, StorageOrder::NHWC>(   \
          {N, S, C}, x, mu, rsig, dy, dgamma, dbeta); \
    }                                                 \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)       \
  template <>                                      \
  void BatchNormTrainingGrad<T, AccT, CPUContext>( \
      const int N,                                 \
      const int C,                                 \
      const int S,                                 \
      const float normalizer,                      \
      const string& data_format,                   \
      const T* x,                                  \
      const AccT* mu,                              \
      const AccT* rsig,                            \
      const AccT* gamma,                           \
      const AccT* dgamma,                          \
      const AccT* dbeta,                           \
      const T* dy,                                 \
      T* dx,                                       \
      CPUContext* ctx) {                           \
    CPU_UNSUPPORTED_DTYPE(T);                      \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                                 \
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
    dbeta = (dgamma == dbeta ? dgamma + C : dbeta);                          \
    if (data_format == "NCHW") {                                             \
      _BatchNormTrainingGrad<T, AccT, StorageOrder::NCHW>(                   \
          {N, C, S}, normalizer, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    } else if (data_format == "NHWC") {                                      \
      _BatchNormTrainingGrad<T, AccT, StorageOrder::NHWC>(                   \
          {N, S, C}, normalizer, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    }                                                                        \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)        \
  template <>                                       \
  void BatchNormInferenceGrad<T, AccT, CPUContext>( \
      const int N,                                  \
      const int C,                                  \
      const int S,                                  \
      const string& data_format,                    \
      const T* x,                                   \
      const AccT* mu,                               \
      const AccT* rsig,                             \
      const AccT* gamma,                            \
      const T* dy,                                  \
      AccT* dgamma,                                 \
      AccT* dbeta,                                  \
      T* dx,                                        \
      CPUContext* ctx) {                            \
    CPU_UNSUPPORTED_DTYPE(T);                       \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                          \
  template <>                                                         \
  void BatchNormInferenceGrad<T, AccT, CPUContext>(                   \
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
      CPUContext* ctx) {                                              \
    if (dgamma != nullptr) {                                          \
      BatchNormWGrad<T, AccT>(                                        \
          N, C, S, data_format, x, mu, rsig, dy, dgamma, dbeta, ctx); \
    }                                                                 \
    if (data_format == "NCHW") {                                      \
      _BatchNormInferenceGrad<T, AccT, StorageOrder::NCHW>(           \
          N, C, S, rsig, gamma, dy, dx);                              \
    } else if (data_format == "NHWC") {                               \
      _BatchNormInferenceGrad<T, AccT, StorageOrder::NHWC>(           \
          N, C, S, rsig, gamma, dy, dx);                              \
    }                                                                 \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
