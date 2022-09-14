#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Tile(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y) {
  const auto N = math::utils::Prod(num_dims, y_dims);
  vec64_t index(num_dims, 0);
  int64_t xi;
  for (int i = 0; i < N; ++i) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += (index[d] % x_dims[d]) * x_strides[d];
    }
    y[i] = x[xi];
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

} // namespace

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

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
