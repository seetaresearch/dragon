#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Assign(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_strides,
    const int64_t* starts,
    const T* x,
    T* y) {
  const auto N = math::utils::Prod(num_dims, x_dims);
  vec64_t index(num_dims, 0);
  int yi;
  for (int xi = 0; xi < N; ++xi) {
    yi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      yi += (index[d] + starts[d]) * y_strides[d];
    }
    y[yi] = x[xi];
    math::utils::IncreaseIndexInDims(num_dims, x_dims, index.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                       \
  template <>                                           \
  void Assign<T, CPUContext>(                           \
      const int num_dims,                               \
      const int64_t* x_dims,                            \
      const int64_t* y_strides,                         \
      const int64_t* starts,                            \
      const T* x,                                       \
      T* y,                                             \
      CPUContext* ctx) {                                \
    _Assign(num_dims, x_dims, y_strides, starts, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
