#include "dragon/core/memory.h"
#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormExpectation(
    const std::array<int, 3>& dims,
    const Tp denorm,
    const Tx* x,
    Tp* ex,
    Tp* ex2) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int count = dims[0] * dims[1] * dims[2];
  std::array<int, 3> idx = {0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const Tx x_val = x[i];
    const int pi = idx[kCDim];
    ex[pi] += x_val;
    ex2[pi] += x_val * x_val;
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
  for (int i = 0; i < dims[kCDim]; ++i) {
    ex[i] = ex[i] * denorm;
    ex2[i] = ex2[i] * denorm;
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormInternalGrad(
    const std::array<int, 3>& dims,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tp* dgamma,
    Tp* dbeta) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int count = dims[0] * dims[1] * dims[2];
  std::array<int, 3> idx = {0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const int pi = idx[kCDim];
    dgamma[pi] += dy[i] * (x[i] - mu[pi]) * rsig[pi];
    dbeta[pi] += dy[i];
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormTrainingGrad(
    const std::array<int, 3>& dims,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tp* dgamma,
    const Tp* dbeta,
    const Tx* dy,
    Tx* dx) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int count = dims[0] * dims[1] * dims[2];
  const Tp denom = Tp(1) / static_cast<Tp>(count / dims[kCDim]);
  std::array<int, 3> idx = {0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const int pi = idx[kCDim];
    const Tp x_norm = (x[i] - mu[pi]) * rsig[pi];
    dx[i] = gamma[pi] * rsig[pi] *
        (dy[i] - (x_norm * dgamma[pi] + dbeta[pi]) * denom);
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormWGrad(
    const std::array<int, 3>& dims,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tx* dy,
    Tp* dgamma,
    Tp* dbeta) {
  const int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int count = dims[0] * dims[1] * dims[2];
  std::array<int, 3> idx = {0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const int pi = idx[kCDim];
    dgamma[pi] += dy[i] * (x[i] - mu[pi]) * rsig[pi];
    dbeta[pi] += dy[i];
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename Tx, typename Tp, StorageOrder kOrder>
void _BatchNormInferenceGrad(
    const int N,
    const int C,
    const int S,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tx* dx) {
  if (kOrder == StorageOrder::NCHW) {
    const int CS = C * S;
    for (int i = 0; i < N; ++i) {
      EigenArrayMap<Tx>(dx + i * CS, S, C) =
          (ConstEigenArrayMap<Tx>(dy + i * CS, S, C).rowwise() *
           (ConstEigenVectorArrayMap<Tp>(gamma, C) *
            ConstEigenVectorArrayMap<Tp>(rsig, C))
               .transpose());
    }
  } else if (kOrder == StorageOrder::NHWC) {
    EigenArrayMap<Tx>(dx, C, N * S) =
        (ConstEigenArrayMap<Tx>(dy, C, N * S).colwise() *
         (ConstEigenVectorArrayMap<Tp>(gamma, C) *
          ConstEigenVectorArrayMap<Tp>(rsig, C)));
  }
}

} //  namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GRAD_KERNEL_LAUNCHER(Tx, Tp)                      \
  template <>                                                    \
  void BatchNormExpectation<Tx, Tp, CPUContext>(                 \
      const int N,                                               \
      const int C,                                               \
      const int S,                                               \
      const Tp denorm,                                           \
      const string& data_format,                                 \
      const Tx* x,                                               \
      Tp* ex,                                                    \
      Tp* ex2,                                                   \
      CPUContext* ctx) {                                         \
    math::Set(C, Tp(0), ex, ctx);                                \
    math::Set(C, Tp(0), ex2, ctx);                               \
    if (data_format == "NCHW") {                                 \
      _BatchNormExpectation<Tx, Tp, StorageOrder::NCHW>(         \
          {N, C, S}, denorm, x, ex, ex2);                        \
    } else if (data_format == "NHWC") {                          \
      _BatchNormExpectation<Tx, Tp, StorageOrder::NHWC>(         \
          {N, S, C}, denorm, x, ex, ex2);                        \
    }                                                            \
  }                                                              \
  template <>                                                    \
  void BatchNormInternalGrad<Tx, Tp, CPUContext>(                \
      const int N,                                               \
      const int C,                                               \
      const int S,                                               \
      const string& data_format,                                 \
      const Tx* x,                                               \
      const Tp* mu,                                              \
      const Tp* rsig,                                            \
      const Tp* gamma,                                           \
      const Tx* dy,                                              \
      Tp* dgamma,                                                \
      Tp* dbeta,                                                 \
      CPUContext* ctx) {                                         \
    math::Set(C, Tp(0), dgamma, ctx);                            \
    math::Set(C, Tp(0), dbeta, ctx);                             \
    if (data_format == "NCHW") {                                 \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NCHW>(        \
          {N, C, S}, x, mu, rsig, gamma, dy, dgamma, dbeta);     \
    } else if (data_format == "NHWC") {                          \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NHWC>(        \
          {N, S, C}, x, mu, rsig, gamma, dy, dgamma, dbeta);     \
    }                                                            \
  }                                                              \
  template <>                                                    \
  void BatchNormTrainingGrad<Tx, Tp, CPUContext>(                \
      const int N,                                               \
      const int C,                                               \
      const int S,                                               \
      const string& data_format,                                 \
      const Tx* x,                                               \
      const Tp* mu,                                              \
      const Tp* rsig,                                            \
      const Tp* gamma,                                           \
      const Tp* dgamma,                                          \
      const Tp* dbeta,                                           \
      const Tx* dy,                                              \
      Tx* dx,                                                    \
      CPUContext* ctx) {                                         \
    if (data_format == "NCHW") {                                 \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NCHW>(        \
          {N, C, S}, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    } else if (data_format == "NHWC") {                          \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NHWC>(        \
          {N, S, C}, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    }                                                            \
  }                                                              \
  template <>                                                    \
  void BatchNormBackwardTraining<Tx, Tp, CPUContext>(            \
      const int N,                                               \
      const int C,                                               \
      const int S,                                               \
      const string& data_format,                                 \
      const Tx* x,                                               \
      const Tp* mu,                                              \
      const Tp* rsig,                                            \
      const Tp* gamma,                                           \
      const Tx* dy,                                              \
      Tx* dx,                                                    \
      Tp* dgamma,                                                \
      Tp* dbeta,                                                 \
      CPUContext* ctx) {                                         \
    math::Set(C, Tp(0), dgamma, ctx);                            \
    math::Set(C, Tp(0), dbeta, ctx);                             \
    if (data_format == "NCHW") {                                 \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NCHW>(        \
          {N, C, S}, x, mu, rsig, gamma, dy, dgamma, dbeta);     \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NCHW>(        \
          {N, C, S}, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    } else if (data_format == "NHWC") {                          \
      _BatchNormInternalGrad<Tx, Tp, StorageOrder::NHWC>(        \
          {N, S, C}, x, mu, rsig, gamma, dy, dgamma, dbeta);     \
      _BatchNormTrainingGrad<Tx, Tp, StorageOrder::NHWC>(        \
          {N, S, C}, x, mu, rsig, gamma, dgamma, dbeta, dy, dx); \
    }                                                            \
  }                                                              \
  template <>                                                    \
  void BatchNormBackwardInference<Tx, Tp, CPUContext>(           \
      const int N,                                               \
      const int C,                                               \
      const int S,                                               \
      const string& data_format,                                 \
      const Tx* x,                                               \
      const Tp* mu,                                              \
      const Tp* rsig,                                            \
      const Tp* gamma,                                           \
      const Tx* dy,                                              \
      Tx* dx,                                                    \
      Tp* dgamma,                                                \
      Tp* dbeta,                                                 \
      CPUContext* ctx) {                                         \
    if (data_format == "NCHW") {                                 \
      if (dgamma != nullptr) {                                   \
        math::Set(C, Tp(0), dgamma, ctx);                        \
        math::Set(C, Tp(0), dbeta, ctx);                         \
        _BatchNormWGrad<Tx, Tp, StorageOrder::NCHW>(             \
            {N, C, S}, x, mu, rsig, dy, dgamma, dbeta);          \
      }                                                          \
      _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NCHW>(       \
          N, C, S, rsig, gamma, dy, dx);                         \
    } else if (data_format == "NHWC") {                          \
      if (dgamma != nullptr) {                                   \
        math::Set(C, Tp(0), dgamma, ctx);                        \
        math::Set(C, Tp(0), dbeta, ctx);                         \
        _BatchNormWGrad<Tx, Tp, StorageOrder::NHWC>(             \
            {N, S, C}, x, mu, rsig, dy, dgamma, dbeta);          \
      }                                                          \
      _BatchNormInferenceGrad<Tx, Tp, StorageOrder::NHWC>(       \
          N, C, S, rsig, gamma, dy, dx);                         \
    }                                                            \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
