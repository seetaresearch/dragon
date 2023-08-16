#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Reverse(
    const int num_dims,
    const uint8_t* x_flips,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y) {
  const auto N = math::utils::Prod(num_dims, y_dims);
  vec64_t index(num_dims, 0);
  for (int64_t yi = 0; yi < N; ++yi) {
    int64_t xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += (x_flips[d] ? y_dims[d] - index[d] - 1 : index[d]) * x_strides[d];
    }
    y[yi] = x[xi];
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                         \
  template <>                                             \
  void Reverse<T, CPUContext>(                            \
      const int num_dims,                                 \
      const uint8_t* x_flips,                             \
      const int64_t* x_strides,                           \
      const int64_t* y_dims,                              \
      const T* x,                                         \
      T* y,                                               \
      CPUContext* ctx) {                                  \
    _Reverse(num_dims, x_flips, x_strides, y_dims, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
