#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _OneHot(
    const int count,
    const int depth,
    const int on_value,
    const T* x,
    T* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    const int val = (int)x[i];
    y[i * depth + val] = (T)on_value;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)          \
  template <>                              \
  void OneHot<T, CPUContext>(              \
      const int count,                     \
      const int depth,                     \
      const int on_value,                  \
      const T* x,                          \
      T* y,                                \
      CPUContext* ctx) {                   \
    _OneHot(count, depth, on_value, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
