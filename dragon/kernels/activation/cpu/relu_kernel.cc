#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Relu(const int N, const float alpha, const T* x, T* y) {
  EigenVectorArrayMap<T> Y(y, N);
  ConstEigenVectorArrayMap<T> X(x, N);
  if (alpha == 0.f) {
    Y = X.cwiseMax(T(0));
  } else {
    Y = X.cwiseMax(T(0)) + X.cwiseMin(T(0)) * T(alpha);
  }
}

template <typename T>
void _ReluN(const int N, const float high, const T* x, T* y) {
  EigenVectorArrayMap<T> Y(y, N);
  ConstEigenVectorArrayMap<T> X(x, N);
  Y = X.cwiseMax(T(0)).cwiseMin(T(high));
}

template <typename T>
void _ReluGrad(const int N, const float alpha, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T> dX(dx, N);
  ConstEigenVectorArrayMap<T> Y(y, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  if (alpha == 0.f) {
    dX = (Y > T(0)).select(dY, T(0));
  } else {
    dX = (Y > T(0)).select(dY, dY * T(alpha));
  }
}

template <typename T>
void _ReluNGrad(const int N, const float high, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T> dX(dx, N);
  ConstEigenVectorArrayMap<T> Y(y, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  dX = (Y > T(0) && Y < T(high)).select(dY, T(0));
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                  \
  template <>                                                            \
  void name<T, CPUContext>(                                              \
      const int N, const float arg, const T* x, T* y, CPUContext* ctx) { \
    using EigenT = typename math::Traits<T>::eigen_type;                 \
    _##name(N, arg, (const EigenT*)x, (EigenT*)y);                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                           \
  template <>                                                          \
  void name<T, CPUContext>(                                            \
      const int N,                                                     \
      const float arg,                                                 \
      const T* dy,                                                     \
      const T* y,                                                      \
      T* dx,                                                           \
      CPUContext* ctx) {                                               \
    using EigenT = typename math::Traits<T>::eigen_type;               \
    _##name(N, arg, (const EigenT*)dy, (const EigenT*)y, (EigenT*)dx); \
  }

DEFINE_KERNEL_LAUNCHER(Relu, float16);
DEFINE_KERNEL_LAUNCHER(Relu, bfloat16);
DEFINE_KERNEL_LAUNCHER(Relu, float);
DEFINE_KERNEL_LAUNCHER(Relu, double);
DEFINE_KERNEL_LAUNCHER(ReluN, float16);
DEFINE_KERNEL_LAUNCHER(ReluN, bfloat16);
DEFINE_KERNEL_LAUNCHER(ReluN, float);
DEFINE_KERNEL_LAUNCHER(ReluN, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
