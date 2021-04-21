#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Gelu(const int N, const T* x, T* y) {
  const T kRsqrt2 = 0.7071067811865475;
  for (int i = 0; i < N; ++i) {
    const T val = x[i];
    y[i] = val * (T(1) + erf(val * kRsqrt2)) * T(0.5);
  }
}

template <>
void _Gelu<float16>(const int N, const float16* x, float16* y) {
  const float kRsqrt2 = 0.7071067811865475;
  for (int i = 0; i < N; ++i) {
    const float val = convert::To<float>(x[i]);
    y[i] = convert::To<float16>(val * (1.f + erf(val * kRsqrt2)) * 0.5f);
  }
}

template <typename T>
void _GeluGrad(const int N, const T* dy, const T* x, T* dx) {
  const T kAlpha = 0.3989422804014327; // 0.5 * Sqrt(2/Pi)
  const T kRsqrt2 = 0.7071067811865475;
  ConstEigenVectorArrayMap<T> dY(dy, N);
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T> dX(dx, N);
  for (int i = 0; i < N; ++i) {
    dx[i] = (T(1) + erf(x[i] * kRsqrt2)) * T(0.5);
  }
  dX = dY * (dX + X * ((T(-0.5) * X.square()).exp() * kAlpha));
}

template <>
void _GeluGrad<float16>(
    const int N,
    const float16* dy,
    const float16* x,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _ApproxGelu(const int N, const T* x, T* y) {
  const T kAlpha = 0.7978845608028654; // Sqrt(2/Pi)
  const T kBeta = 0.035677408136300125; // Sqrt(2/Pi) * 0.044715
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T> Y(y, N);
  Y = X * ((X * kAlpha + X.cube() * kBeta).tanh() + T(1)) * T(0.5);
}

template <>
void _ApproxGelu<float16>(const int N, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _ApproxGeluGrad(const int N, const T* dy, const T* x, T* dx) {
  const T kAlpha = 0.7978845608028654; // Sqrt(2/Pi)
  const T kBeta = 0.035677408136300125; // Sqrt(2/Pi) * 0.044715
  const T kGamma = 0.10703222440890037; // Sqrt(2/Pi) * 0.044715 * 3
  ConstEigenVectorArrayMap<T> dY(dy, N);
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T> Y(dx, N);
  EigenVectorArrayMap<T> dX(dx, N);
  Y = (X * kAlpha + X.cube() * kBeta).tanh();
  dX = T(0.5) * dY *
      (T(1) + Y + (X - X * Y.square()) * (kGamma * X.square() + kAlpha));
}

template <>
void _ApproxGeluGrad<float16>(
    const int N,
    const float16* dy,
    const float16* x,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                      \
  template <>                                                                \
  void name<T, CPUContext>(const int N, const T* x, T* y, CPUContext* ctx) { \
    _##name(N, x, y);                                                        \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                         \
  void name<T, CPUContext>(                                           \
      const int N, const T* dy, const T* x, T* dx, CPUContext* ctx) { \
    _##name(N, dy, x, dx);                                            \
  }

DEFINE_KERNEL_LAUNCHER(Gelu, float16);
DEFINE_KERNEL_LAUNCHER(Gelu, float);
DEFINE_KERNEL_LAUNCHER(Gelu, double);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float16);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, double);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
