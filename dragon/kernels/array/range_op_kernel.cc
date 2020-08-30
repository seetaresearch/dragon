#include "dragon/utils/cast.h"
#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Range(const int count, const float start, const float delta, T* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    y[i] = static_cast<T>(start + i * delta);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Range<float16, CPUContext>(
    const int count,
    const float start,
    const float delta,
    float16* y,
    CPUContext* ctx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    y[i] = cast::to<float16>(start + (float)i * delta);
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)   \
  template <>                       \
  void Range<T, CPUContext>(        \
      const int count,              \
      const float start,            \
      const float delta,            \
      T* y,                         \
      CPUContext* ctx) {            \
    _Range(count, start, delta, y); \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
