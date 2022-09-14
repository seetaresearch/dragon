#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Elu(const int N, const T alpha, const T* x, T* y) {
  EigenVectorArrayMap<T>(y, N) =
      ConstEigenVectorArrayMap<T>(x, N).unaryExpr([&](T a) {
        return a > T(0) ? a : alpha * (std::exp(std::min(a, T(0))) - T(1));
      });
}

template <>
void _Elu<float16>(
    const int N,
    const float16 alpha,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _EluGrad(const int N, const T alpha, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      ConstEigenVectorArrayMap<T>(y, N).unaryExpr(
          [&](T a) { return a > T(0) ? T(1) : alpha + a; });
}

template <>
void _EluGrad<float16>(
    const int N,
    const float16 alpha,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // EluGrad

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void Elu<T, CPUContext>(                                                 \
      const int N, const float alpha, const T* x, T* y, CPUContext* ctx) { \
    _Elu(N, convert::To<T>(alpha), x, y);                                  \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)             \
  template <>                                      \
  void EluGrad<T, CPUContext>(                     \
      const int N,                                 \
      const float alpha,                           \
      const T* dy,                                 \
      const T* y,                                  \
      T* dx,                                       \
      CPUContext* ctx) {                           \
    _EluGrad(N, convert::To<T>(alpha), dy, y, dx); \
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
