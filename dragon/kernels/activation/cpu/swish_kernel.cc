#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Silu(const int N, const T* x, T* y) {
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(y, N) = X / (T(1) + (-X).exp());
}

template <typename T>
void _HardSwish(const int N, const T* x, T* y) {
  const T kAlpha = 0.1666666666666667;
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(y, N) =
      X * ((X * kAlpha + T(0.5)).cwiseMin(T(1)).cwiseMax(T(0)));
}

template <>
void _Silu<float16>(const int N, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void _HardSwish<float16>(const int N, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _SiluGrad(const int N, const T* dy, const T* x, T* dx) {
  ConstEigenVectorArrayMap<T> X(x, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  EigenVectorArrayMap<T> dX(dx, N);
  dX = T(1) / (T(1) + (-X).exp());
  dX = dY * dX * (X + T(1) - X * dX);
}

template <>
void _SiluGrad<float16>(
    const int N,
    const float16* dy,
    const float16* x,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // SiluGrad

template <typename T>
void _HardSwishGrad(const int N, const T* dy, const T* x, T* dx) {
  const T kAlpha2 = 0.3333333333333333;
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      ConstEigenVectorArrayMap<T>(x, N).unaryExpr([&](T a) {
        return a < T(-3) ? T(0) : (a < T(3) ? a * kAlpha2 + T(0.5) : T(1));
      });
}

template <>
void _HardSwishGrad<float16>(
    const int N,
    const float16* dy,
    const float16* x,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // HardSwishGrad

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

DEFINE_KERNEL_LAUNCHER(Silu, float16);
DEFINE_KERNEL_LAUNCHER(Silu, float);
DEFINE_KERNEL_LAUNCHER(Silu, double);
DEFINE_KERNEL_LAUNCHER(HardSwish, float16);
DEFINE_KERNEL_LAUNCHER(HardSwish, float);
DEFINE_KERNEL_LAUNCHER(HardSwish, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
