#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _L1Norm(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* x,
    T* y) {
  const auto CxS = C * S;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      const auto offset = i * CxS + j;
      EigenStridedVectorMap<T> Y(y + offset, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorMap<T> X(x + offset, 1, C, EigenInnerStride(S));
      const auto norm = float(X.template lpNorm<1>()) / normalizer;
      Y = X / T(std::max(norm, epsilon));
    }
  }
}

template <typename T>
void _L2Norm(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* x,
    T* y) {
  const auto CxS = C * S;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      const auto offset = i * CxS + j;
      EigenStridedVectorMap<T> Y(y + offset, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorMap<T> X(x + offset, 1, C, EigenInnerStride(S));
      const auto norm = std::sqrt(float(X.squaredNorm()) / normalizer);
      Y = X / T(std::max(norm, epsilon));
    }
  }
}

template <typename T>
void _L1NormGrad(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float eps,
    const T* dy,
    const T* x,
    T* dx) {
  const auto CxS = C * S;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      const auto offset = i * CxS + j;
      EigenStridedVectorMap<T> dX(dx + offset, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorMap<T> dY(dy + offset, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorMap<T> X(x + offset, 1, C, EigenInnerStride(S));
      auto norm = std::max(float(X.template lpNorm<1>()) / normalizer, eps);
      auto norm2 = norm * norm, sum = float(dY.dot(X)) / normalizer;
      dX = dY / T(norm) - X.array().sign().matrix() / T(norm2 / sum);
    }
  }
}

template <typename T>
void _L2NormGrad(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float eps,
    const T* dy,
    const T* x,
    T* dx) {
  const auto CxS = C * S;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      const auto offset = i * CxS + j;
      EigenStridedVectorMap<T> dX(dx + offset, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorMap<T> dY(dy + offset, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorMap<T> X(x + offset, 1, C, EigenInnerStride(S));
      auto norm = std::max(std::sqrt(float(X.squaredNorm()) / normalizer), eps);
      auto norm3 = std::pow(norm, 3.f), sum = float(dY.dot(X)) / normalizer;
      dX = dY / T(norm) - X / T(norm3 / sum);
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                              \
  template <>                                                        \
  void name<T, CPUContext>(                                          \
      const int N,                                                   \
      const int S,                                                   \
      const int C,                                                   \
      const float normalizer,                                        \
      const float eps,                                               \
      const T* x,                                                    \
      T* y,                                                          \
      CPUContext* ctx) {                                             \
    using EigenT = math::Traits<T>::eigen_type;                      \
    _##name(N, S, C, normalizer, eps, (const EigenT*)x, (EigenT*)y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                      \
  template <>                                                     \
  void name<T, CPUContext>(                                       \
      const int N,                                                \
      const int S,                                                \
      const int C,                                                \
      const float normalizer,                                     \
      const float eps,                                            \
      const T* dy,                                                \
      const T* x,                                                 \
      T* dx,                                                      \
      CPUContext* ctx) {                                          \
    _##name(                                                      \
        N,                                                        \
        S,                                                        \
        C,                                                        \
        normalizer,                                               \
        eps,                                                      \
        reinterpret_cast<const math::Traits<T>::eigen_type*>(dy), \
        reinterpret_cast<const math::Traits<T>::eigen_type*>(x),  \
        reinterpret_cast<math::Traits<T>::eigen_type*>(dx));      \
  }

DEFINE_KERNEL_LAUNCHER(L1Norm, float16);
DEFINE_KERNEL_LAUNCHER(L1Norm, bfloat16);
DEFINE_KERNEL_LAUNCHER(L1Norm, float);
DEFINE_KERNEL_LAUNCHER(L1Norm, double);
DEFINE_KERNEL_LAUNCHER(L2Norm, float16);
DEFINE_KERNEL_LAUNCHER(L2Norm, bfloat16);
DEFINE_KERNEL_LAUNCHER(L2Norm, float);
DEFINE_KERNEL_LAUNCHER(L2Norm, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
