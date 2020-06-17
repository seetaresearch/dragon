#include "dragon/utils/cast.h"
#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Selu(const int count, const T alpha, const T gamma, const T* x, T* y) {
  const T alphaXgamma = alpha * gamma;
  EigenVectorArrayMap<T>(y, count) =
      ConstEigenVectorArrayMap<T>(x, count).unaryExpr([&](T a) {
        return a > T(0) ? gamma * a
                        : alphaXgamma * ((std::exp(std::min(a, T(0))) - T(1)));
      });
}

template <>
void _Selu<float16>(
    const int count,
    const float16 alpha,
    const float16 gamma,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _SeluGrad(
    const int count,
    const T alpha,
    const T gamma,
    const T* dy,
    const T* y,
    T* dx) {
  const T alphaXgamma = alpha * gamma;
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      ConstEigenVectorArrayMap<T>(y, count).unaryExpr(
          [&](T a) { return a > T(0) ? gamma : (alphaXgamma + a); });
}

template <>
void _SeluGrad<float16>(
    const int count,
    const float16 alpha,
    const float16 gamma,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // SeluGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                               \
  template <>                                                   \
  void Selu<T, CPUContext>(                                     \
      const int count,                                          \
      const float alpha,                                        \
      const float gamma,                                        \
      const T* x,                                               \
      T* y,                                                     \
      CPUContext* ctx) {                                        \
    _Selu(count, cast::to<T>(alpha), cast::to<T>(gamma), x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                   \
  template <>                                                            \
  void SeluGrad<T, CPUContext>(                                          \
      const int count,                                                   \
      const float alpha,                                                 \
      const float gamma,                                                 \
      const T* dy,                                                       \
      const T* y,                                                        \
      T* dx,                                                             \
      CPUContext* tx) {                                                  \
    _SeluGrad(count, cast::to<T>(alpha), cast::to<T>(gamma), dy, y, dx); \
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
