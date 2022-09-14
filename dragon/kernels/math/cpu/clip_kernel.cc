#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Clip(const int N, const T low, const T high, const T* x, T* y) {
  EigenVectorMap<T>(y, N) =
      ConstEigenVectorMap<T>(x, N).cwiseMax(low).cwiseMin(high);
}

template <>
void _Clip<float16>(
    const int N,
    const float16 low,
    const float16 high,
    const float16* x,
    float16* y) {
  auto lowf = convert::To<float>(low);
  auto highf = convert::To<float>(high);
  for (int i = 0; i < N; ++i) {
    auto val = convert::To<float>(x[i]);
    val = std::max(lowf, std::min(val, highf));
    y[i] = convert::To<float16>(val);
  }
}

template <typename T>
void _ClipGrad(
    const int N,
    const T low,
    const T high,
    const T* dy,
    const T* x,
    T* dx) {
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(dx, N) =
      (X < low || X > high).select(T(0), ConstEigenVectorArrayMap<T>(dy, N));
}

template <>
void _ClipGrad<float16>(
    const int N,
    const float16 low,
    const float16 high,
    const float16* dy,
    const float16* x,
    float16* dx) {
  auto lowf = convert::To<float>(low);
  auto highf = convert::To<float>(high);
  auto kZero = convert::To<float16>(0.f);
  for (int i = 0; i < N; ++i) {
    auto val = convert::To<float>(x[i]);
    dx[i] = (val < lowf || val > highf) ? kZero : dy[i];
  }
} // ClipGrad

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                              \
  template <>                                                  \
  void Clip<T, CPUContext>(                                    \
      const int N,                                             \
      const float low,                                         \
      const float high,                                        \
      const T* x,                                              \
      T* y,                                                    \
      CPUContext* ctx) {                                       \
    _Clip(N, convert::To<T>(low), convert::To<T>(high), x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void ClipGrad<T, CPUContext>(                                         \
      const int N,                                                      \
      const float low,                                                  \
      const float high,                                                 \
      const T* dy,                                                      \
      const T* x,                                                       \
      T* dx,                                                            \
      CPUContext* ctx) {                                                \
    _ClipGrad(N, convert::To<T>(low), convert::To<T>(high), dy, x, dx); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
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
