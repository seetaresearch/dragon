#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _SetEye(const int n, const int m, const int k, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i * m + k + i] = convert::To<T>(1.f);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void Eye<T, CPUContext>(                                            \
      const int n, const int m, const int k, T* y, CPUContext* ctx) { \
    math::Set(n* m, convert::To<T>(0.f), y, ctx);                     \
    if (k > 0) {                                                      \
      if (m - k > 0) _SetEye(m - k, m, k, y);                         \
    } else {                                                          \
      if (n + k > 0) _SetEye(n + k, m, 0, y - k * m);                 \
    }                                                                 \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
