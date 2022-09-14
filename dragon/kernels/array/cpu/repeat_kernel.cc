#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Repeat(
    const int N,
    const int S,
    const int C,
    const int repeats,
    const T* x,
    T* y,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      for (int k = 0; k < repeats; ++k) {
        math::Copy(S, x, y, ctx);
        y += S;
      }
      x += S;
    }
  }
}

template <typename T>
void _RepeatGrad(
    const int N,
    const int S,
    const int C,
    const int repeats,
    const T* dy,
    T* dx,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      math::Copy(S, dy, dx, ctx);
      dy += S;
      for (int k = 1; k < repeats; ++k) {
        math::Add(S, dy, dx, dx, ctx);
        dy += S;
      }
      dx += S;
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)   \
  template <>                             \
  void name<T, CPUContext>(               \
      const int N,                        \
      const int S,                        \
      const int C,                        \
      const int repeats,                  \
      const T* x,                         \
      T* y,                               \
      CPUContext* ctx) {                  \
    _##name(N, S, C, repeats, x, y, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(Repeat, bool);
DEFINE_KERNEL_LAUNCHER(Repeat, uint8_t);
DEFINE_KERNEL_LAUNCHER(Repeat, int8_t);
DEFINE_KERNEL_LAUNCHER(Repeat, int);
DEFINE_KERNEL_LAUNCHER(Repeat, int64_t);
DEFINE_KERNEL_LAUNCHER(Repeat, float16);
DEFINE_KERNEL_LAUNCHER(Repeat, float);
DEFINE_KERNEL_LAUNCHER(Repeat, double);
DEFINE_KERNEL_LAUNCHER(RepeatGrad, float16);
DEFINE_KERNEL_LAUNCHER(RepeatGrad, float);
DEFINE_KERNEL_LAUNCHER(RepeatGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
