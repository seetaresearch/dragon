#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _IndexSelect(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int num_indices,
    const int64_t* indices,
    const T* x,
    T* y,
    CPUContext* ctx) {
  int index;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < num_indices; ++j) {
      index = indices[j];
      index = index >= 0 ? index : index + axis_dim;
      const T* offset_x = x + (i * axis_dim + index) * inner_dim;
      math::Copy(inner_dim, offset_x, y, ctx);
      y += inner_dim;
    }
  }
}

template <typename T>
void _IndexSelectGrad(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int num_indices,
    const int64_t* indices,
    const T* dy,
    T* dx,
    CPUContext* ctx) {
  int index;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < num_indices; ++j) {
      index = indices[j];
      index = index >= 0 ? index : index + axis_dim;
      T* offset_dx = dx + (i * axis_dim + index) * inner_dim;
      math::Add(inner_dim, dy, offset_dx, offset_dx, ctx);
      dy += inner_dim;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                                       \
  template <>                                                                 \
  void name<T, CPUContext>(                                                   \
      const int outer_dim,                                                    \
      const int inner_dim,                                                    \
      const int axis_dim,                                                     \
      const int num_indices,                                                  \
      const int64_t* indices,                                                 \
      const T* x,                                                             \
      T* y,                                                                   \
      CPUContext* ctx) {                                                      \
    _##name(outer_dim, inner_dim, axis_dim, num_indices, indices, x, y, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(IndexSelect, bool);
DEFINE_KERNEL_LAUNCHER(IndexSelect, int8_t);
DEFINE_KERNEL_LAUNCHER(IndexSelect, uint8_t);
DEFINE_KERNEL_LAUNCHER(IndexSelect, int);
DEFINE_KERNEL_LAUNCHER(IndexSelect, int64_t);
DEFINE_KERNEL_LAUNCHER(IndexSelect, float16);
DEFINE_KERNEL_LAUNCHER(IndexSelect, float);
DEFINE_KERNEL_LAUNCHER(IndexSelect, double);

DEFINE_KERNEL_LAUNCHER(IndexSelectGrad, int8_t);
DEFINE_KERNEL_LAUNCHER(IndexSelectGrad, uint8_t);
DEFINE_KERNEL_LAUNCHER(IndexSelectGrad, int);
DEFINE_KERNEL_LAUNCHER(IndexSelectGrad, int64_t);
DEFINE_KERNEL_LAUNCHER(IndexSelectGrad, float16);
DEFINE_KERNEL_LAUNCHER(IndexSelectGrad, float);
DEFINE_KERNEL_LAUNCHER(IndexSelectGrad, double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
