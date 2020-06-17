#include "dragon/utils/cast.h"
#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Clip(const int count, const T low, const T high, const T* x, T* y) {
  EigenVectorMap<T>(y, count) =
      ConstEigenVectorMap<T>(x, count).cwiseMax(low).cwiseMin(high);
}

template <>
void _Clip<float16>(
    const int count,
    const float16 low,
    const float16 high,
    const float16* x,
    float16* y) {
  auto lowf = cast::to<float>(low);
  auto highf = cast::to<float>(high);
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    auto val = cast::to<float>(x[i]);
    val = std::max(lowf, std::min(val, highf));
    y[i] = cast::to<float16>(val);
  }
}

template <typename T>
void _ClipGrad(
    const int count,
    const T low,
    const T high,
    const T* dy,
    const T* x,
    T* dx) {
  ConstEigenVectorArrayMap<T> X(x, count);
  EigenVectorArrayMap<T>(dx, count) =
      (X < low || X > high)
          .select(T(0), ConstEigenVectorArrayMap<T>(dy, count));
}

template <>
void _ClipGrad<float16>(
    const int count,
    const float16 low,
    const float16 high,
    const float16* dy,
    const float16* x,
    float16* dx) {
  auto lowf = cast::to<float>(low);
  auto highf = cast::to<float>(high);
  auto kZero = cast::to<float16>(0.f);
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    auto val = cast::to<float>(x[i]);
    dx[i] = (val < lowf || val > highf) ? kZero : dy[i];
  }
} // ClipGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                            \
  template <>                                                \
  void Clip<T, CPUContext>(                                  \
      const int count,                                       \
      const float low,                                       \
      const float high,                                      \
      const T* x,                                            \
      T* y,                                                  \
      CPUContext* ctx) {                                     \
    _Clip(count, cast::to<T>(low), cast::to<T>(high), x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void ClipGrad<T, CPUContext>(                                       \
      const int count,                                                \
      const float low,                                                \
      const float high,                                               \
      const T* dy,                                                    \
      const T* x,                                                     \
      T* dx,                                                          \
      CPUContext* ctx) {                                              \
    _ClipGrad(count, cast::to<T>(low), cast::to<T>(high), dy, x, dx); \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
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

} // namespace kernel

} // namespace dragon
