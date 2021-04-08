#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Swish(const int N, const T* x, T* y) {
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(y, N) = X / (T(1) + (-X).exp());
}

template <>
void _Swish<float16>(const int N, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _SwishGrad(const int N, const T* dy, const T* x, const T* y, T* dx) {
  ConstEigenVectorArrayMap<T> X(x, N);
  ConstEigenVectorArrayMap<T> Y(y, N);
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      (Y + (T(1) / (T(1) + (-X).exp())) * (T(1) - Y));
}

template <>
void _SwishGrad<float16>(
    const int N,
    const float16* dy,
    const float16* x,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void Swish<T, CPUContext>(const int N, const T* x, T* y, CPUContext* ctx) { \
    _Swish(N, x, y);                                                          \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T) \
  template <>                          \
  void SwishGrad<T, CPUContext>(       \
      const int N,                     \
      const T* dy,                     \
      const T* x,                      \
      const T* y,                      \
      T* dx,                           \
      CPUContext* ctx) {               \
    _SwishGrad(N, dy, x, y, dx);       \
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
