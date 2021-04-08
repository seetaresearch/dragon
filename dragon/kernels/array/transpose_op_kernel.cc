#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Transpose(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y) {
  const auto N =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int64_t xi;
  for (int yi = 0; yi < N; ++yi) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += index[d] * x_strides[d];
    }
    y[yi] = x[xi];
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

template <typename T>
void _TransposeGrad(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* dy,
    T* dx) {
  const auto N =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int64_t xi;
  for (int yi = 0; yi < N; ++yi) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += index[d] * x_strides[d];
    }
    dx[xi] = dy[yi];
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)         \
  template <>                                   \
  void name<T, CPUContext>(                     \
      const int num_dims,                       \
      const int64_t* x_strides,                 \
      const int64_t* y_dims,                    \
      const T* x,                               \
      T* y,                                     \
      CPUContext* ctx) {                        \
    _##name(num_dims, x_strides, y_dims, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(Transpose, bool);
DEFINE_KERNEL_LAUNCHER(Transpose, uint8_t);
DEFINE_KERNEL_LAUNCHER(Transpose, int8_t);
DEFINE_KERNEL_LAUNCHER(Transpose, int);
DEFINE_KERNEL_LAUNCHER(Transpose, int64_t);
DEFINE_KERNEL_LAUNCHER(Transpose, float16);
DEFINE_KERNEL_LAUNCHER(Transpose, float);
DEFINE_KERNEL_LAUNCHER(Transpose, double);
DEFINE_KERNEL_LAUNCHER(TransposeGrad, float16);
DEFINE_KERNEL_LAUNCHER(TransposeGrad, float);
DEFINE_KERNEL_LAUNCHER(TransposeGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
