#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _ScatterElements(
    const int axis,
    const int num_dims,
    const T value,
    const int64_t* dims,
    const int64_t* y_strides,
    const int64_t* index,
    T* y) {
  const auto N = math::utils::Prod(num_dims, dims);
  vec64_t dim_index(num_dims, 0);
  for (int64_t i = 0; i < N; ++i) {
    int64_t yi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      yi += (d == axis ? index[i] : dim_index[d]) * y_strides[d];
    }
    y[yi] = value;
    math::utils::IncreaseIndexInDims(num_dims, dims, dim_index.data());
  }
}

template <typename T>
void _ScatterElements(
    const int axis,
    const int num_dims,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const T* x,
    T* y) {
  const auto N = math::utils::Prod(num_dims, dims);
  vec64_t dim_index(num_dims, 0);
  for (int64_t i = 0; i < N; ++i) {
    int64_t xi = 0, yi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += dim_index[d] * x_strides[d];
      yi += (d == axis ? index[i] : dim_index[d]) * y_strides[d];
    }
    y[yi] = x[xi];
    math::utils::IncreaseIndexInDims(num_dims, dims, dim_index.data());
  }
}

template <typename T, typename AccT>
void _ScatterAdd(
    const int axis,
    const int num_dims,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const T* x,
    AccT* y) {
  const auto N = math::utils::Prod(num_dims, dims);
  vec64_t dim_index(num_dims, 0);
  for (int64_t i = 0; i < N; ++i) {
    int64_t xi = 0, yi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += dim_index[d] * x_strides[d];
      yi += (d == axis ? index[i] : dim_index[d]) * y_strides[d];
    }
    y[yi] += convert::To<AccT>(x[xi]);
    math::utils::IncreaseIndexInDims(num_dims, dims, dim_index.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                        \
  template <>                                                  \
  void name<T, CPUContext>(                                    \
      const int axis,                                          \
      const int num_dims,                                      \
      const T value,                                           \
      const int64_t* dims,                                     \
      const int64_t* y_strides,                                \
      const int64_t* index,                                    \
      T* y,                                                    \
      CPUContext* ctx) {                                       \
    _##name(axis, num_dims, value, dims, y_strides, index, y); \
  }

DEFINE_KERNEL_LAUNCHER(ScatterElements, bool);
DEFINE_KERNEL_LAUNCHER(ScatterElements, uint8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int64_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, bfloat16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float);
DEFINE_KERNEL_LAUNCHER(ScatterElements, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void name<T, CPUContext>(                                           \
      const int axis,                                                 \
      const int num_dims,                                             \
      const int64_t* dims,                                            \
      const int64_t* x_strides,                                       \
      const int64_t* y_strides,                                       \
      const int64_t* index,                                           \
      const T* x,                                                     \
      T* y,                                                           \
      CPUContext* ctx) {                                              \
    _##name(axis, num_dims, dims, x_strides, y_strides, index, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(ScatterElements, bool);
DEFINE_KERNEL_LAUNCHER(ScatterElements, uint8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int64_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, bfloat16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float);
DEFINE_KERNEL_LAUNCHER(ScatterElements, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T, AccT)                         \
  template <>                                                         \
  void name<T, AccT, CPUContext>(                                     \
      const int axis,                                                 \
      const int num_dims,                                             \
      const int64_t* dims,                                            \
      const int64_t* x_strides,                                       \
      const int64_t* y_strides,                                       \
      const int64_t* index,                                           \
      const T* x,                                                     \
      AccT* y,                                                        \
      CPUContext* ctx) {                                              \
    _##name(axis, num_dims, dims, x_strides, y_strides, index, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(ScatterAdd, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, int, int)
DEFINE_KERNEL_LAUNCHER(ScatterAdd, int64_t, int64_t)
DEFINE_KERNEL_LAUNCHER(ScatterAdd, float16, float);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, bfloat16, float);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, float, float)
DEFINE_KERNEL_LAUNCHER(ScatterAdd, double, float);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
