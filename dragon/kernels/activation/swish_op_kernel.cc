#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Swish(const int count, const T* x, T* y) {
  ConstEigenVectorArrayMap<T> X(x, count);
  EigenVectorArrayMap<T>(y, count) = X / (T(1) + (-X).exp());
}

template <>
void _Swish<float16>(const int count, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _SwishGrad(const int count, const T* dy, const T* x, const T* y, T* dx) {
  ConstEigenVectorArrayMap<T> X(x, count);
  ConstEigenVectorArrayMap<T> Y(y, count);
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      (Y + (T(1) / (T(1) + (-X).exp())) * (T(1) - Y));
}

template <>
void _SwishGrad<float16>(
    const int count,
    const float16* dy,
    const float16* x,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                           \
  template <>                                               \
  void Swish<T, CPUContext>(                                \
      const int count, const T* x, T* y, CPUContext* ctx) { \
    _Swish(count, x, y);                                    \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T) \
  template <>                          \
  void SwishGrad<T, CPUContext>(       \
      const int count,                 \
      const T* dy,                     \
      const T* x,                      \
      const T* y,                      \
      T* dx,                           \
      CPUContext* ctx) {               \
    _SwishGrad(count, dy, x, y, dx);   \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
