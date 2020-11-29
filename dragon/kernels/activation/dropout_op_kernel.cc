#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _ApplyMask(
    const int count,
    const T scale,
    const T* x,
    const uint8_t* mask,
    T* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    y[i] = x[i] * (T)mask[i] * scale;
  }
}

template <>
void _ApplyMask<float16>(
    const int count,
    const float16 scale,
    const float16* x,
    const uint8_t* mask,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _Dropout(
    const int count,
    const T ratio,
    const T scale,
    const T* x,
    uint8_t* mask,
    T* y,
    CPUContext* ctx) {
  math::RandomBernoulli(count, T(1) - ratio, mask, ctx);
  _ApplyMask(count, scale, x, mask, y);
}

template <>
void _Dropout<float16>(
    const int count,
    const float16 ratio,
    const float16 scale,
    const float16* x,
    uint8_t* mask,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void ApplyMask<T, CPUContext>(                                               \
      const int count,                                                         \
      const float scale,                                                       \
      const T* x,                                                              \
      const uint8_t* mask,                                                     \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    _ApplyMask(count, convert::To<T>(scale), x, mask, y);                      \
  }                                                                            \
  template <>                                                                  \
  void Dropout<T, CPUContext>(                                                 \
      const int count,                                                         \
      const float ratio,                                                       \
      const float scale,                                                       \
      const T* x,                                                              \
      uint8_t* mask,                                                           \
      T* y,                                                                    \
      uint32_t* r,                                                             \
      CPUContext* ctx) {                                                       \
    _Dropout(                                                                  \
        count, convert::To<T>(ratio), convert::To<T>(scale), x, mask, y, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
