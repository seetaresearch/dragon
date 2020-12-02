#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Slice(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* starts,
    const T* x,
    T* y) {
  const auto count =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int xi;
  for (int yi = 0; yi < count; ++yi) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += (index[d] + starts[d]) * x_strides[d];
    }
    y[yi] = x[xi];
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

template <typename T>
void _SliceGrad(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* starts,
    const T* dy,
    T* dx) {
  const auto count =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int xi;
  for (int yi = 0; yi < count; ++yi) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += (index[d] + starts[d]) * x_strides[d];
    }
    dx[xi] = dy[yi];
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                 \
  template <>                                           \
  void name<T, CPUContext>(                             \
      const int num_dims,                               \
      const int64_t* x_strides,                         \
      const int64_t* y_dims,                            \
      const int64_t* starts,                            \
      const T* x,                                       \
      T* y,                                             \
      CPUContext* ctx) {                                \
    _##name(num_dims, x_strides, y_dims, starts, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(Slice, bool);
DEFINE_KERNEL_LAUNCHER(Slice, int8_t);
DEFINE_KERNEL_LAUNCHER(Slice, uint8_t);
DEFINE_KERNEL_LAUNCHER(Slice, int);
DEFINE_KERNEL_LAUNCHER(Slice, int64_t);
DEFINE_KERNEL_LAUNCHER(Slice, float16);
DEFINE_KERNEL_LAUNCHER(Slice, float);
DEFINE_KERNEL_LAUNCHER(Slice, double);
DEFINE_KERNEL_LAUNCHER(SliceGrad, bool);
DEFINE_KERNEL_LAUNCHER(SliceGrad, int8_t);
DEFINE_KERNEL_LAUNCHER(SliceGrad, uint8_t);
DEFINE_KERNEL_LAUNCHER(SliceGrad, int);
DEFINE_KERNEL_LAUNCHER(SliceGrad, int64_t);
DEFINE_KERNEL_LAUNCHER(SliceGrad, float16);
DEFINE_KERNEL_LAUNCHER(SliceGrad, float);
DEFINE_KERNEL_LAUNCHER(SliceGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
