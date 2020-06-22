#include "dragon/utils/math_functions.h"
#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _DropPath(
    const int rows,
    const int cols,
    const float scale,
    const T* x,
    const float* mask,
    T* y) {
  auto count = rows * cols;
  auto thresh = 1.f - (1.f / scale);
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    y[i] = x[i] * (T)(mask[i / cols] > thresh) * scale;
  }
}

template <>
void _DropPath<float16>(
    const int rows,
    const int cols,
    const float scale,
    const float16* x,
    const float* mask,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)             \
  template <>                                 \
  void DropPath<T, CPUContext>(               \
      const int rows,                         \
      const int cols,                         \
      const float scale,                      \
      const T* x,                             \
      const float* mask,                      \
      T* y,                                   \
      CPUContext* ctx) {                      \
    _DropPath(rows, cols, scale, x, mask, y); \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
