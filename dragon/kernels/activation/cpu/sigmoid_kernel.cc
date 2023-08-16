#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

#define DEFINE_KERNEL_LAUNCHER(T)                            \
  template <>                                                \
  void Sigmoid<T, CPUContext>(                               \
      const int N, const T* x, T* y, CPUContext* ctx) {      \
    using EigenT = math::Traits<T>::eigen_type;              \
    EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);            \
    ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N); \
    Y = EigenT(1) / (EigenT(1) + (-X).exp());                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void SigmoidGrad<T, CPUContext>(                                    \
      const int N, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);                   \
    ConstEigenVectorArrayMap<EigenT> Y((const EigenT*)y, N);          \
    ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);        \
    dX = dY * Y * (EigenT(1) - Y);                                    \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void HardSigmoid<T, CPUContext>(                                    \
      const int N,                                                    \
      const float alpha,                                              \
      const float beta,                                               \
      const T* x,                                                     \
      T* y,                                                           \
      CPUContext* ctx) {                                              \
    using EigenT = math::Traits<T>::eigen_type;                       \
    const auto kAlpha = EigenT(alpha), kBeta = EigenT(beta);          \
    EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);                     \
    ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);          \
    Y = (X * kAlpha + kBeta).cwiseMin(EigenT(1)).cwiseMax(EigenT(0)); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                    \
  template <>                                                             \
  void HardSigmoidGrad<T, CPUContext>(                                    \
      const int N,                                                        \
      const float alpha,                                                  \
      const T* dy,                                                        \
      const T* y,                                                         \
      T* dx,                                                              \
      CPUContext* ctx) {                                                  \
    using EigenT = math::Traits<T>::eigen_type;                           \
    const auto kAlpha = EigenT(alpha);                                    \
    EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);                       \
    ConstEigenVectorArrayMap<EigenT> Y((const EigenT*)y, N);              \
    ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);            \
    dX = (Y > EigenT(0) && Y < EigenT(1)).select(dY * kAlpha, EigenT(0)); \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
