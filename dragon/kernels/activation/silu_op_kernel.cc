#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Silu(const int N, const T* x, T* y) {
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(y, N) = X / (T(1) + (-X).exp());
}

template <>
void _Silu<float16>(const int N, const float16* x, float16* y) {
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
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void Silu<T, CPUContext>(const int N, const T* x, T* y, CPUContext* ctx) { \
    _Silu(N, x, y);                                                          \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void SiluGrad<T, CPUContext>(                                       \
      const int N, const T* dy, const T* x, T* dx, CPUContext* ctx) { \
    _SiluGrad(N, dy, x, dx);                                          \
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
