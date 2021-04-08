#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _L1Normalize(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const T normalizer,
    const T epsilon,
    const T* x,
    T* y) {
  const auto dim = reduce_dim * inner_dim;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      auto offset = i * dim + j;
      auto X = ConstEigenStridedVectorMap<T>(
          x + offset, 1, reduce_dim, EigenInnerStride(inner_dim));
      EigenStridedVectorMap<T>(
          y + offset, 1, reduce_dim, EigenInnerStride(inner_dim)) =
          X / std::max(X.template lpNorm<1>() / normalizer, epsilon);
    }
  }
}

template <typename T>
void _L2Normalize(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const T normalizer,
    const T epsilon,
    const T* x,
    T* y) {
  const auto dim = reduce_dim * inner_dim;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      auto offset = i * dim + j;
      auto X = ConstEigenStridedVectorMap<T>(
          x + offset, 1, reduce_dim, EigenInnerStride(inner_dim));
      EigenStridedVectorMap<T>(
          y + offset, 1, reduce_dim, EigenInnerStride(inner_dim)) =
          X / std::max(std::sqrt(X.squaredNorm() / normalizer), epsilon);
    }
  }
}

template <typename T>
void _L1NormalizeGrad(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const T normalizer,
    const T epsilon,
    const T* dy,
    const T* x,
    T* dx) {
  const auto dim = reduce_dim * inner_dim;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      auto offset = i * dim + j;
      auto dY = ConstEigenStridedVectorMap<T>(
          dy + offset, 1, reduce_dim, EigenInnerStride(inner_dim));
      auto X = ConstEigenStridedVectorMap<T>(
          x + offset, 1, reduce_dim, EigenInnerStride(inner_dim));
      auto norm = std::max(X.template lpNorm<1>() / normalizer, epsilon);
      auto norm2 = std::pow(norm, T(2));
      EigenStridedVectorMap<T>(
          dx + offset, 1, reduce_dim, EigenInnerStride(inner_dim)) =
          (dY / norm) -
          (X.array().sign().matrix() / norm2) * dY.dot(X) / normalizer;
    }
  }
}

template <typename T>
void _L2NormalizeGrad(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const T normalizer,
    const T epsilon,
    const T* dy,
    const T* x,
    T* dx) {
  const auto dim = reduce_dim * inner_dim;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      auto offset = i * dim + j;
      auto dY = ConstEigenStridedVectorMap<T>(
          dy + offset, 1, reduce_dim, EigenInnerStride(inner_dim));
      auto X = ConstEigenStridedVectorMap<T>(
          x + offset, 1, reduce_dim, EigenInnerStride(inner_dim));
      auto norm = std::max(std::sqrt(X.squaredNorm() / normalizer), epsilon);
      auto norm3 = std::pow(norm, T(3));
      EigenStridedVectorMap<T>(
          dx + offset, 1, reduce_dim, EigenInnerStride(inner_dim)) =
          (dY / norm) - ((X / norm3) * dY.dot(X) / normalizer);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void L1Normalize<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const float normalizer,
    const float epsilon,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void L2Normalize<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const float normalizer,
    const float epsilon,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void L1NormalizeGrad<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const float normalizer,
    const float epsilon,
    const float16* dy,
    const float16* x,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // L1NormalizeGrad

template <>
void L2NormalizeGrad<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int reduce_dim,
    const float normalizer,
    const float epsilon,
    const float16* dy,
    const float16* x,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // L2NormalizeGrad

#define DEFINE_KERNEL_LAUNCHER(name, T)                                  \
  template <>                                                            \
  void name<T, CPUContext>(                                              \
      const int outer_dim,                                               \
      const int inner_dim,                                               \
      const int reduce_dim,                                              \
      const float normalizer,                                            \
      const float eps,                                                   \
      const T* x,                                                        \
      T* y,                                                              \
      CPUContext* ctx) {                                                 \
    _##name<T>(outer_dim, inner_dim, reduce_dim, normalizer, eps, x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                  \
  template <>                                                                 \
  void name<T, CPUContext>(                                                   \
      const int outer_dim,                                                    \
      const int inner_dim,                                                    \
      const int reduce_dim,                                                   \
      const float normalizer,                                                 \
      const float eps,                                                        \
      const T* dy,                                                            \
      const T* x,                                                             \
      T* dx,                                                                  \
      CPUContext* ctx) {                                                      \
    _##name<T>(outer_dim, inner_dim, reduce_dim, normalizer, eps, dy, x, dx); \
  }

DEFINE_KERNEL_LAUNCHER(L1Normalize, float);
DEFINE_KERNEL_LAUNCHER(L1Normalize, double);
DEFINE_KERNEL_LAUNCHER(L2Normalize, float);
DEFINE_KERNEL_LAUNCHER(L2Normalize, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
