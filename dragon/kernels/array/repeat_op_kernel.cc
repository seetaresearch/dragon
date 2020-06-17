#include "dragon/utils/math_functions.h"
#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Repeat(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int repeats,
    const T* x,
    T* y,
    CPUContext* ctx) {
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < axis_dim; ++j) {
      for (int k = 0; k < repeats; ++k) {
        math::Copy(inner_dim, x, y, ctx);
        y += inner_dim;
      }
      x += inner_dim;
    }
  }
}

template <typename T>
void _RepeatGrad(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int repeats,
    const T* dy,
    T* dx,
    CPUContext* ctx) {
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < axis_dim; ++j) {
      math::Copy(inner_dim, dy, dx, ctx);
      dy += inner_dim;
      for (int k = 1; k < repeats; ++k) {
        math::Add(inner_dim, dy, dx, dx, ctx);
        dy += inner_dim;
      }
      dx += inner_dim;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                    \
  void name<T, CPUContext>(                                      \
      const int outer_dim,                                       \
      const int inner_dim,                                       \
      const int axis_dim,                                        \
      const int repeats,                                         \
      const T* x,                                                \
      T* y,                                                      \
      CPUContext* ctx) {                                         \
    _##name(outer_dim, inner_dim, axis_dim, repeats, x, y, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(Repeat, bool);
DEFINE_KERNEL_LAUNCHER(Repeat, int8_t);
DEFINE_KERNEL_LAUNCHER(Repeat, uint8_t);
DEFINE_KERNEL_LAUNCHER(Repeat, int);
DEFINE_KERNEL_LAUNCHER(Repeat, int64_t);
DEFINE_KERNEL_LAUNCHER(Repeat, float16);
DEFINE_KERNEL_LAUNCHER(Repeat, float);
DEFINE_KERNEL_LAUNCHER(Repeat, double);

DEFINE_KERNEL_LAUNCHER(RepeatGrad, float16);
DEFINE_KERNEL_LAUNCHER(RepeatGrad, float);
DEFINE_KERNEL_LAUNCHER(RepeatGrad, double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
