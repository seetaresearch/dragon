#include "dragon/core/memory.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _GroupNormFusedParams(
    const int N,
    const int G,
    const int D,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  const int C = G * D;
  ConstEigenArrayMap<T> gamma_arr(gamma, D, G);
  ConstEigenArrayMap<T> beta_arr(beta, D, G);
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<T> scale_arr(scale + i * C, D, G);
    scale_arr = gamma_arr.rowwise() *
        ConstEigenVectorArrayMap<T>(rsig + i * G, G).transpose();
    EigenArrayMap<T>(bias + i * C, D, G) = beta_arr -
        scale_arr.rowwise() *
            ConstEigenVectorArrayMap<T>(mu + i * G, G).transpose();
  }
}

template <typename T, typename AccT>
void _GroupNormNCHW(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  EigenArrayMap<T>(y, S, N * C) =
      (ConstEigenArrayMap<T>(x, S, N * C).rowwise() *
       ConstEigenVectorArrayMap<AccT>(scale, N * C).transpose())
          .rowwise() +
      ConstEigenVectorArrayMap<AccT>(bias, N * C).transpose();
}

template <typename T, typename AccT>
void _GroupNormNHWC(
    const int N,
    const int C,
    const int S,
    const T* x,
    const AccT* scale,
    const AccT* bias,
    T* y) {
  const int SC = S * C;
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<T>(y + i * SC, C, S) =
        (ConstEigenArrayMap<T>(x + i * SC, C, S).colwise() *
         ConstEigenVectorArrayMap<AccT>(scale + i * C, C))
            .colwise() +
        ConstEigenVectorArrayMap<AccT>(bias + i * C, C);
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
  const int count = dims[0] * dims[1] * dims[2] * dims[3];
  std::array<int, 4> idx = {0, 0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const int mi = idx[0] * dims[kGDim] + idx[kGDim];
    const int gi = idx[kGDim] * dims[kDDim] + idx[kDDim];
    ds[mi] += gamma[gi] * dy[i] * x[i];
    db[mi] += gamma[gi] * dy[i];
    math::utils::IncreaseIndexInDims(4, dims.data(), idx.data());
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
  const int count = dims[0] * dims[1] * dims[2] * dims[3];
  const int S = kOrder == StorageOrder::NCHW ? dims[3] : dims[1];
  const AccT denom = AccT(1) / static_cast<AccT>(dims[kDDim] * S);
  std::array<int, 4> idx = {0, 0, 0, 0};
  for (int i = 0; i < count; ++i) {
    const int mi = idx[0] * dims[kGDim] + idx[kGDim];
    const int gi = idx[kGDim] * dims[kDDim] + idx[kDDim];
    const AccT u = (db[mi] * mu[mi] - ds[mi]) * (x[i] - mu[mi]) *
        math::utils::Cube(rsig[mi]);
    const AccT v = db[mi] * rsig[mi];
    dx[i] = gamma[gi] * dy[i] * rsig[mi] + (u - v) * denom;
    dgamma[gi] += dy[i] * (x[i] - mu[mi]) * rsig[mi];
    dbeta[gi] += dy[i];
    math::utils::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void GroupNorm<float16, float, CPUContext>(
    const int N,
    const int G,
    const int D,
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
void GroupNormGrad<float16, float, CPUContext>(
    const int N,
    const int G,
    const int D,
    const int S,
    const string& data_format,
    const float16* x,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float16* dy,
    float* ds,
    float* db,
    float* dgamma,
    float* dbeta,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // GroupNormBackward

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                                 \
  template <>                                                           \
  void GroupNorm<T, AccT, CPUContext>(                                  \
      const int N,                                                      \
      const int G,                                                      \
      const int D,                                                      \
      const int S,                                                      \
      const string& data_format,                                        \
      const T* x,                                                       \
      const AccT* mu,                                                   \
      const AccT* rsig,                                                 \
      const AccT* gamma,                                                \
      const AccT* beta,                                                 \
      AccT* scale,                                                      \
      AccT* bias,                                                       \
      T* y,                                                             \
      CPUContext* ctx) {                                                \
    const int C = G * D;                                                \
    _GroupNormFusedParams(N, G, D, mu, rsig, gamma, beta, scale, bias); \
    if (data_format == "NCHW") {                                        \
      _GroupNormNCHW(N, C, S, x, scale, bias, y);                       \
    } else if (data_format == "NHWC") {                                 \
      _GroupNormNHWC(N, C, S, x, scale, bias, y);                       \
    }                                                                   \
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
    math::Set(N* G, AccT(0), ds, ctx);                                      \
    math::Set(N* G, AccT(0), db, ctx);                                      \
    math::Set(G* D, AccT(0), dgamma, ctx);                                  \
    math::Set(G* D, AccT(0), dbeta, ctx);                                   \
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

DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
