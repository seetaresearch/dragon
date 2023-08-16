#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT, StorageOrder kOrder>
void _GroupNorm(
    const std::array<int, 4>& dims,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* beta,
    T* y) {
  const int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
  const int NxGxDxS = dims[0] * dims[1] * dims[2] * dims[3];
  std::array<int, 4> index = {0, 0, 0, 0};
  for (int i = 0; i < NxGxDxS; ++i) {
    const int ng = index[0] * dims[kGDim] + index[kGDim];
    const int c = index[kGDim] * dims[kDDim] + index[kDDim];
    AccT val = (convert::To<AccT>(x[i]) - mu[ng]) * rsig[ng];
    y[i] = convert::To<T>(val * gamma[c] + beta[c]);
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
void _GroupNormInternalGrad(
    const std::array<int, 4>& dims,
    const T* x,
    const AccT* gamma,
    const T* dy,
    AccT* ds,
    AccT* db) {
  const int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
  const int NxGxDxS = dims[0] * dims[1] * dims[2] * dims[3];
  std::array<int, 4> index = {0, 0, 0, 0};
  for (int i = 0; i < NxGxDxS; ++i) {
    const int ng = index[0] * dims[kGDim] + index[kGDim];
    const int c = index[kGDim] * dims[kDDim] + index[kDDim];
    ds[ng] += gamma[c] * dy[i] * x[i];
    db[ng] += gamma[c] * dy[i];
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
void _GroupNormGrad(
    const std::array<int, 4>& dims,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* ds,
    const AccT* db,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta,
    T* dx) {
  const int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  const int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
  const int NxGxDxS = dims[0] * dims[1] * dims[2] * dims[3];
  const int S = kOrder == StorageOrder::NCHW ? dims[3] : dims[1];
  const AccT denom = AccT(1) / static_cast<AccT>(dims[kDDim] * S);
  std::array<int, 4> index = {0, 0, 0, 0};
  for (int i = 0; i < NxGxDxS; ++i) {
    const int ng = index[0] * dims[kGDim] + index[kGDim];
    const int c = index[kGDim] * dims[kDDim] + index[kDDim];
    const AccT u = (db[ng] * mu[ng] - ds[ng]) * (x[i] - mu[ng]) *
        math::utils::Cube(rsig[ng]);
    const AccT v = db[ng] * rsig[ng];
    dx[i] = gamma[c] * dy[i] * rsig[ng] + (u - v) * denom;
    dgamma[c] += dy[i] * (x[i] - mu[ng]) * rsig[ng];
    dbeta[c] += dy[i];
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T, AccT)               \
  template <>                                         \
  void GroupNorm<T, AccT, CPUContext>(                \
      const int N,                                    \
      const int G,                                    \
      const int D,                                    \
      const int S,                                    \
      const string& data_format,                      \
      const T* x,                                     \
      const AccT* mu,                                 \
      const AccT* rsig,                               \
      const AccT* gamma,                              \
      const AccT* beta,                               \
      T* y,                                           \
      CPUContext* ctx) {                              \
    if (data_format == "NCHW") {                      \
      _GroupNorm<T, AccT, StorageOrder::NCHW>(        \
          {N, G, D, S}, x, mu, rsig, gamma, beta, y); \
    } else if (data_format == "NHWC") {               \
      _GroupNorm<T, AccT, StorageOrder::NHWC>(        \
          {N, S, G, D}, x, mu, rsig, gamma, beta, y); \
    }                                                 \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                                \
  template <>                                                               \
  void GroupNormGrad<T, AccT, CPUContext>(                                  \
      const int N,                                                          \
      const int G,                                                          \
      const int D,                                                          \
      const int S,                                                          \
      const string& data_format,                                            \
      const T* x,                                                           \
      const AccT* mu,                                                       \
      const AccT* rsig,                                                     \
      const AccT* gamma,                                                    \
      const T* dy,                                                          \
      AccT* ds,                                                             \
      AccT* db,                                                             \
      AccT* dgamma,                                                         \
      AccT* dbeta,                                                          \
      T* dx,                                                                \
      CPUContext* ctx) {                                                    \
    db = ds == db ? ds + N * G : db;                                        \
    math::Set((N * G), AccT(0), ds, ctx);                                   \
    math::Set((N * G), AccT(0), db, ctx);                                   \
    math::Set((G * D), AccT(0), dgamma, ctx);                               \
    math::Set((G * D), AccT(0), dbeta, ctx);                                \
    if (data_format == "NCHW") {                                            \
      _GroupNormInternalGrad<T, AccT, StorageOrder::NCHW>(                  \
          {N, G, D, S}, x, gamma, dy, ds, db);                              \
      _GroupNormGrad<T, AccT, StorageOrder::NCHW>(                          \
          {N, G, D, S}, x, mu, rsig, gamma, ds, db, dy, dgamma, dbeta, dx); \
    } else if (data_format == "NHWC") {                                     \
      _GroupNormInternalGrad<T, AccT, StorageOrder::NHWC>(                  \
          {N, S, G, D}, x, gamma, dy, ds, db);                              \
      _GroupNormGrad<T, AccT, StorageOrder::NHWC>(                          \
          {N, S, G, D}, x, mu, rsig, gamma, ds, db, dy, dgamma, dbeta, dx); \
    }                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT) \
  template <>                                \
  void GroupNormGrad<T, AccT, CPUContext>(   \
      const int N,                           \
      const int G,                           \
      const int D,                           \
      const int S,                           \
      const string& data_format,             \
      const T* x,                            \
      const AccT* mu,                        \
      const AccT* rsig,                      \
      const AccT* gamma,                     \
      const T* dy,                           \
      AccT* ds,                              \
      AccT* db,                              \
      AccT* dgamma,                          \
      AccT* dbeta,                           \
      T* dx,                                 \
      CPUContext* ctx) {                     \
    CPU_UNSUPPORTED_DTYPE(T);                \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
