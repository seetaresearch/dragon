#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Tile(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y) {
  const auto count =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int64_t xi;
  for (int i = 0; i < count; ++i) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += (index[d] % x_dims[d]) * x_strides[d];
    }
    y[i] = x[xi];
    utils::math::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

template <typename T>
void _TileGrad(
    const int rows,
    const int cols,
    const int multiple,
    const T* dy,
    T* dx,
    CPUContext* ctx) {
  for (int i = 0; i < rows; ++i) {
    math::Copy(cols, dy, dx, ctx);
    dy += cols;
    for (int j = 1; j < multiple; ++j) {
      math::Add(cols, dy, dx, dx, ctx);
      dy += cols;
    }
    dx += cols;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                     \
  template <>                                         \
  void Tile<T, CPUContext>(                           \
      const int num_dims,                             \
      const int64_t* x_dims,                          \
      const int64_t* x_strides,                       \
      const int64_t* y_dims,                          \
      const T* x,                                     \
      T* y,                                           \
      CPUContext* ctx) {                              \
    _Tile(num_dims, x_dims, x_strides, y_dims, x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)            \
  template <>                                     \
  void TileGrad<T, CPUContext>(                   \
      const int rows,                             \
      const int cols,                             \
      const int multiple,                         \
      const T* dy,                                \
      T* dx,                                      \
      CPUContext* ctx) {                          \
    _TileGrad(rows, cols, multiple, dy, dx, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
