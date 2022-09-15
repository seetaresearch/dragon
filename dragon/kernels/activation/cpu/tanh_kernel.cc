#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Tanh(const int N, const T* x, T* y) {
  EigenVectorArrayMap<T>(y, N) = ConstEigenVectorArrayMap<T>(x, N).tanh();
}

template <>
void _Tanh<float16>(const int N, const float16* x, float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _TanhGrad(const int N, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      ConstEigenVectorArrayMap<T>(y, N).unaryExpr(
          [](T a) { return (T(1) - a * a); });
}

template <>
void _TanhGrad<float16>(
    const int N,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // TanhGrad

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void Tanh<T, CPUContext>(const int N, const T* x, T* y, CPUContext* ctx) { \
    _Tanh(N, x, y);                                                          \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void TanhGrad<T, CPUContext>(                                       \
      const int N, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    _TanhGrad(N, dy, y, dx);                                          \
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
