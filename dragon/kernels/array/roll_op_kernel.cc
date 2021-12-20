#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Roll(
    const int num_dims,
    const int64_t* x_shifts,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y) {
  const auto N = math::utils::Prod(num_dims, y_dims);
  vec64_t index(num_dims, 0);
  for (int yi = 0; yi < N; ++yi) {
    int64_t xi = 0, r;
    for (int d = num_dims - 1; d >= 0; --d) {
      r = index[d] - x_shifts[d];
      r = (r < 0 ? r + y_dims[d] : r) % y_dims[d];
      xi += r * x_strides[d];
    }
    y[yi] = x[xi];
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                       \
  template <>                                           \
  void Roll<T, CPUContext>(                             \
      const int num_dims,                               \
      const int64_t* x_shifts,                          \
      const int64_t* x_strides,                         \
      const int64_t* y_dims,                            \
      const T* x,                                       \
      T* y,                                             \
      CPUContext* ctx) {                                \
    _Roll(num_dims, x_shifts, x_strides, y_dims, x, y); \
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

} // namespace kernels

} // namespace dragon
