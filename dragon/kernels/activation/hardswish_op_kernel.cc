#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _HardSwish(
    const int count,
    const T alpha,
    const T beta,
    const T* x,
    T* y) {
  ConstEigenVectorArrayMap<T> X(x, count);
  EigenVectorArrayMap<T>(y, count) =
      X * ((X * alpha + beta).cwiseMin(T(1)).cwiseMax(T(0)));
}

template <>
void _HardSwish<float16>(
    const int count,
    const float16 alpha,
    const float16 beta,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _HardSwishGrad(
    const int count,
    const T alpha,
    const T beta,
    const T* dy,
    const T* x,
    T* dx) {
  const auto bound = beta / alpha;
  const auto alpha2x = alpha * T(2);
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      ConstEigenVectorArrayMap<T>(x, count).unaryExpr([&](T a) {
        return (a < -bound) ? T(0) : (a < bound ? a * alpha2x + beta : T(1));
      });
}

template <>
void _HardSwishGrad<float16>(
    const int count,
    const float16 alpha,
    const float16 beta,
    const float16* dy,
    const float16* x,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void HardSwish<T, CPUContext>(                                          \
      const int count,                                                    \
      const float alpha,                                                  \
      const float beta,                                                   \
      const T* x,                                                         \
      T* y,                                                               \
      CPUContext* ctx) {                                                  \
    _HardSwish(count, convert::To<T>(alpha), convert::To<T>(beta), x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void HardSwishGrad<T, CPUContext>(                                    \
      const int count,                                                  \
      const float alpha,                                                \
      const float beta,                                                 \
      const T* dy,                                                      \
      const T* x,                                                       \
      T* dx,                                                            \
      CPUContext* ctx) {                                                \
    _HardSwishGrad(                                                     \
        count, convert::To<T>(alpha), convert::To<T>(beta), dy, x, dx); \
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
