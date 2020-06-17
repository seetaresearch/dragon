#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Concat(
    const int outer_dim,
    const int inner_dim,
    const int x_axis_dim,
    const int y_axis_dim,
    const int index,
    const T* x,
    T* y) {
  const int offset = index * inner_dim;
  const int x_cols = x_axis_dim * inner_dim;
  const int y_cols = y_axis_dim * inner_dim;
  for (int i = 0; i < outer_dim; ++i) {
    std::memcpy(y + i * y_cols + offset, x + i * x_cols, x_cols * sizeof(T));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                       \
  template <>                                                           \
  void Concat<T, CPUContext>(                                           \
      const int outer_dim,                                              \
      const int inner_dim,                                              \
      const int x_axis_dim,                                             \
      const int y_axis_dim,                                             \
      const int index,                                                  \
      const T* x,                                                       \
      T* y,                                                             \
      CPUContext* ctx) {                                                \
    _Concat(outer_dim, inner_dim, x_axis_dim, y_axis_dim, index, x, y); \
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
