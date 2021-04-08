#include "dragon/utils/conversions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SetOneHot(const int N, const int depth, const T value, const T* x, T* y) {
  for (int i = 0; i < N; ++i) {
    y[i * depth + int(x[i])] = value;
  }
}

template <>
void _SetOneHot<float16>(
    const int N,
    const int depth,
    const float16 value,
    const float16* x,
    float16* y) {
  for (int i = 0; i < N; ++i) {
    y[i * depth + int(convert::To<float>(x[i]))] = value;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                      \
  template <>                                          \
  void SetOneHot<T, CPUContext>(                       \
      const int N,                                     \
      const int depth,                                 \
      const float value,                               \
      const T* x,                                      \
      T* y,                                            \
      CPUContext* ctx) {                               \
    _SetOneHot(N, depth, convert::To<T>(value), x, y); \
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
