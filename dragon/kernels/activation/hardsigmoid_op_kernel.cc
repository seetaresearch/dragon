#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _HardSigmoid(
    const int count,
    const T alpha,
    const T beta,
    const T* x,
    T* y) {
  EigenVectorArrayMap<T>(y, count) =
      (ConstEigenVectorArrayMap<T>(x, count) * alpha + beta)
          .cwiseMin(T(1))
          .cwiseMax(T(0));
}

template <>
void _HardSigmoid<float16>(
    const int count,
    const float16 alpha,
    const float16 beta,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _HardSigmoidGrad(
    const int count,
    const T alpha,
    const T* dy,
    const T* y,
    T* dx) {
  ConstEigenVectorArrayMap<T> Y(y, count);
  EigenVectorArrayMap<T>(dx, count) =
      (Y > T(0) && Y < T(1))
          .select(ConstEigenVectorArrayMap<T>(dy, count) * alpha, T(0));
}

template <>
void _HardSigmoidGrad<float16>(
    const int count,
    const float16 alpha,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void HardSigmoid<T, CPUContext>(                                          \
      const int count,                                                      \
      const float alpha,                                                    \
      const float beta,                                                     \
      const T* x,                                                           \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    _HardSigmoid(count, convert::To<T>(alpha), convert::To<T>(beta), x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                         \
  template <>                                                  \
  void HardSigmoidGrad<T, CPUContext>(                         \
      const int count,                                         \
      const float alpha,                                       \
      const T* dy,                                             \
      const T* y,                                              \
      T* dx,                                                   \
      CPUContext* ctx) {                                       \
    _HardSigmoidGrad(count, convert::To<T>(alpha), dy, y, dx); \
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
