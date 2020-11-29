#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Tanh(const int count, const T* x, T* y) {
  EigenVectorArrayMap<T>(y, count) =
      ConstEigenVectorArrayMap<T>(x, count).tanh();
}

template <>
void _Tanh<float16>(const int count, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _TanhGrad(const int count, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      ConstEigenVectorArrayMap<T>(y, count).unaryExpr(
          [](T a) { return (T(1) - a * a); });
}

template <>
void _TanhGrad<float16>(
    const int count,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // TanhGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                           \
  template <>                                               \
  void Tanh<T, CPUContext>(                                 \
      const int count, const T* x, T* y, CPUContext* ctx) { \
    _Tanh(count, x, y);                                     \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                    \
  template <>                                                             \
  void TanhGrad<T, CPUContext>(                                           \
      const int count, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    _TanhGrad(count, dy, y, dx);                                          \
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
