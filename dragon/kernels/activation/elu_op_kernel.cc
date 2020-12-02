#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Elu(const int count, const T alpha, const T* x, T* y) {
  EigenVectorArrayMap<T>(y, count) =
      ConstEigenVectorArrayMap<T>(x, count).unaryExpr([&](T a) {
        return a > T(0) ? a : alpha * (std::exp(std::min(a, T(0))) - T(1));
      });
}

template <>
void _Elu<float16>(
    const int count,
    const float16 alpha,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _EluGrad(const int count, const T alpha, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      ConstEigenVectorArrayMap<T>(y, count).unaryExpr(
          [&](T a) { return a > T(0) ? T(1) : alpha + a; });
}

template <>
void _EluGrad<float16>(
    const int count,
    const float16 alpha,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // EluGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void Elu<T, CPUContext>(                                                     \
      const int count, const float alpha, const T* x, T* y, CPUContext* ctx) { \
    _Elu(count, convert::To<T>(alpha), x, y);                                  \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                 \
  template <>                                          \
  void EluGrad<T, CPUContext>(                         \
      const int count,                                 \
      const float alpha,                               \
      const T* dy,                                     \
      const T* y,                                      \
      T* dx,                                           \
      CPUContext* ctx) {                               \
    _EluGrad(count, convert::To<T>(alpha), dy, y, dx); \
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
