#include "dragon/utils/conversions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _RowwiseLinSpace(
    const int rows,
    const int cols,
    const double* start,
    const double* stop,
    T* y) {
  for (int i = 0; i < cols; ++i) {
    const auto delta = (stop[i] - start[i]) / double(rows - 1);
    y[i] = convert::To<T>(start[i]);
    if (rows > 1) {
      y[i + (rows - 1) * cols] = convert::To<T>(stop[i]);
    }
    for (int j = 1; j < rows - 1; ++j) {
      y[i + j * cols] = convert::To<T>(start[i] + double(j) * delta);
    }
  }
}

template <typename T>
void _ColwiseLinSpace(
    const int rows,
    const int cols,
    const double* start,
    const double* stop,
    T* y) {
  for (int i = 0; i < rows; ++i) {
    const auto delta = (stop[i] - start[i]) / double(cols - 1);
    auto* offset_y = y + i * cols;
    offset_y[0] = convert::To<T>(start[i]);
    if (cols > 1) {
      offset_y[cols - 1] = convert::To<T>(stop[i]);
    }
    for (int j = 1; j < cols - 1; ++j) {
      offset_y[j] = convert::To<T>(start[i] + double(j) * delta);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                  \
  template <>                                      \
  void LinSpace<T, CPUContext>(                    \
      const int rows,                              \
      const int cols,                              \
      const int axis,                              \
      const double* start,                         \
      const double* end,                           \
      T* y,                                        \
      CPUContext* ctx) {                           \
    if (axis == 0) {                               \
      _RowwiseLinSpace(rows, cols, start, end, y); \
    } else {                                       \
      _ColwiseLinSpace(rows, cols, start, end, y); \
    }                                              \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
