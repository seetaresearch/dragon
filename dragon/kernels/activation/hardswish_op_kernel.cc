#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _HardSwish(const int N, const T* x, T* y) {
  const T kAlpha = 0.1666666666666667;
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(y, N) =
      X * ((X * kAlpha + T(0.5)).cwiseMin(T(1)).cwiseMax(T(0)));
}

template <>
void _HardSwish<float16>(const int N, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

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
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                       \
  template <>                                           \
  void HardSwish<T, CPUContext>(                        \
      const int N, const T* x, T* y, CPUContext* ctx) { \
    _HardSwish(N, x, y);                                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void HardSwishGrad<T, CPUContext>(                                  \
      const int N, const T* dy, const T* x, T* dx, CPUContext* ctx) { \
    _HardSwishGrad(N, dy, x, dx);                                     \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
