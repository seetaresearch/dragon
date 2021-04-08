#include "dragon/utils/conversions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Range(const int N, const double start, const double delta, T* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = convert::To<T>(start + double(i) * delta);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T) \
  template <>                     \
  void Range<T, CPUContext>(      \
      const int N,                \
      const double start,         \
      const double delta,         \
      T* y,                       \
      CPUContext* ctx) {          \
    _Range(N, start, delta, y);   \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
