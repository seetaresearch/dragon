#include "dragon/utils/conversions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _RowwiseLinSpace(
    const int N,
    const int C,
    const double* starts,
    const double* stops,
    T* y) {
  for (int i = 0; i < C; ++i) {
    const auto delta = (stops[i] - starts[i]) / double(N - 1);
    y[i] = convert::To<T>(starts[i]);
    if (N > 1) {
      y[i + (N - 1) * C] = convert::To<T>(stops[i]);
    }
    for (int j = 1; j < N - 1; ++j) {
      y[i + j * C] = convert::To<T>(starts[i] + double(j) * delta);
    }
  }
}

template <typename T>
void _ColwiseLinSpace(
    const int N,
    const int C,
    const double* starts,
    const double* stops,
    T* y) {
  for (int i = 0; i < N; ++i) {
    const auto delta = (stops[i] - starts[i]) / double(C - 1);
    auto* offset_y = y + i * C;
    offset_y[0] = convert::To<T>(starts[i]);
    if (C > 1) {
      offset_y[C - 1] = convert::To<T>(stops[i]);
    }
    for (int j = 1; j < C - 1; ++j) {
      offset_y[j] = convert::To<T>(starts[i] + double(j) * delta);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)               \
  template <>                                   \
  void LinSpace<T, CPUContext>(                 \
      const int N,                              \
      const int C,                              \
      const int axis,                           \
      const double* starts,                     \
      const double* stops,                      \
      T* y,                                     \
      CPUContext* ctx) {                        \
    if (axis == 0) {                            \
      _RowwiseLinSpace(N, C, starts, stops, y); \
    } else {                                    \
      _ColwiseLinSpace(N, C, starts, stops, y); \
    }                                           \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
