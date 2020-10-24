#include "dragon/utils/cast.h"
#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Relu(const int count, const T alpha, const T* x, T* y) {
  ConstEigenVectorArrayMap<T> X(x, count);
  EigenVectorArrayMap<T>(y, count) =
      X.cwiseMax(T(0)) + X.cwiseMin(T(0)) * alpha;
}

template <>
void _Relu<float16>(
    const int count,
    const float16 alpha,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _ReluN(const int count, const T max_value, const T* x, T* y) {
  EigenVectorMap<T>(y, count) =
      ConstEigenVectorMap<T>(x, count).cwiseMax(T(0)).cwiseMin(max_value);
}

template <>
void _ReluN<float16>(
    const int count,
    const float16 max_value,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _ReluGrad(const int count, const T alpha, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      ConstEigenVectorArrayMap<T>(y, count).unaryExpr(
          [&](T a) { return a > T(0) ? T(1) : alpha; });
}

template <>
void _ReluGrad<float16>(
    const int count,
    const float16 alpha,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // ReluGrad

template <typename T>
void _ReluNGrad(
    const int count,
    const T max_value,
    const T* dy,
    const T* y,
    T* dx) {
  ConstEigenVectorArrayMap<T> Y(y, count);
  EigenVectorArrayMap<T>(dx, count) =
      (Y > T(0) && Y < max_value)
          .select(ConstEigenVectorArrayMap<T>(dy, count), T(0));
}

template <>
void _ReluNGrad<float16>(
    const int count,
    const float16 max_value,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // ReluNGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void Relu<T, CPUContext>(                                                    \
      const int count, const float alpha, const T* x, T* y, CPUContext* ctx) { \
    _Relu(count, cast::to<T>(alpha), x, y);                                    \
  }                                                                            \
  template <>                                                                  \
  void ReluN<T, CPUContext>(                                                   \
      const int count,                                                         \
      const float max_value,                                                   \
      const T* x,                                                              \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    _ReluN(count, cast::to<T>(max_value), x, y);                               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                    \
  template <>                                             \
  void ReluGrad<T, CPUContext>(                           \
      const int count,                                    \
      const float alpha,                                  \
      const T* dy,                                        \
      const T* y,                                         \
      T* dx,                                              \
      CPUContext* ctx) {                                  \
    _ReluGrad(count, cast::to<T>(alpha), dy, y, dx);      \
  }                                                       \
  template <>                                             \
  void ReluNGrad<T, CPUContext>(                          \
      const int count,                                    \
      const float max_value,                              \
      const T* dy,                                        \
      const T* y,                                         \
      T* dx,                                              \
      CPUContext* ctx) {                                  \
    _ReluNGrad(count, cast::to<T>(max_value), dy, y, dx); \
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
