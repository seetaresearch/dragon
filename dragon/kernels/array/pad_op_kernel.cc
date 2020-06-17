#include "dragon/utils/cast.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _ConstPad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T value,
    const T* x,
    T* y) {
  const auto count =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int64_t xi, d, r;
  for (int yi = 0; yi < count; ++yi) {
    xi = 0;
    for (d = num_dims - 1; d >= 0; --d) {
      r = index[d] - pads[d];
      if (r < 0 || r >= x_dims[d]) break;
      xi += r * x_strides[d];
    }
    y[yi] = d >= 0 ? value : x[xi];
    utils::math::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

template <typename T>
void _ReflectPad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T* x,
    T* y) {
  const auto count =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int64_t xi, d, r;
  for (int yi = 0; yi < count; ++yi) {
    xi = 0;
    for (d = num_dims - 1; d >= 0; --d) {
      r = index[d] - pads[d];
      r = std::max(r, -r);
      r = std::min(r, 2 * x_dims[d] - r - 2);
      xi += r * x_strides[d];
    }
    y[yi] = x[xi];
    utils::math::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

template <typename T>
void _EdgePad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T* x,
    T* y) {
  const auto count =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int64_t xi, d, r;
  for (int yi = 0; yi < count; ++yi) {
    xi = 0;
    for (d = num_dims - 1; d >= 0; --d) {
      r = std::min(x_dims[d] - 1, std::max(index[d] - pads[d], int64_t(0)));
      xi += r * x_strides[d];
    }
    y[yi] = x[xi];
    utils::math::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                       \
  template <>                                                 \
  void name<T, CPUContext>(                                   \
      const int num_dims,                                     \
      const int64_t* x_dims,                                  \
      const int64_t* x_strides,                               \
      const int64_t* y_dims,                                  \
      const int64_t* pads,                                    \
      const T* x,                                             \
      T* y,                                                   \
      CPUContext* ctx) {                                      \
    _##name(num_dims, x_dims, x_strides, y_dims, pads, x, y); \
  }

#define DEFINE_CONST_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                 \
  void ConstPad<T, CPUContext>(                                               \
      const int num_dims,                                                     \
      const int64_t* x_dims,                                                  \
      const int64_t* x_strides,                                               \
      const int64_t* y_dims,                                                  \
      const int64_t* pads,                                                    \
      const float value,                                                      \
      const T* x,                                                             \
      T* y,                                                                   \
      CPUContext* ctx) {                                                      \
    _ConstPad(                                                                \
        num_dims, x_dims, x_strides, y_dims, pads, cast::to<T>(value), x, y); \
  }

DEFINE_CONST_KERNEL_LAUNCHER(bool);
DEFINE_CONST_KERNEL_LAUNCHER(int8_t);
DEFINE_CONST_KERNEL_LAUNCHER(uint8_t);
DEFINE_CONST_KERNEL_LAUNCHER(int);
DEFINE_CONST_KERNEL_LAUNCHER(int64_t);
DEFINE_CONST_KERNEL_LAUNCHER(float16);
DEFINE_CONST_KERNEL_LAUNCHER(float);
DEFINE_CONST_KERNEL_LAUNCHER(double);

DEFINE_KERNEL_LAUNCHER(ReflectPad, bool);
DEFINE_KERNEL_LAUNCHER(ReflectPad, int8_t);
DEFINE_KERNEL_LAUNCHER(ReflectPad, uint8_t);
DEFINE_KERNEL_LAUNCHER(ReflectPad, int);
DEFINE_KERNEL_LAUNCHER(ReflectPad, int64_t);
DEFINE_KERNEL_LAUNCHER(ReflectPad, float16);
DEFINE_KERNEL_LAUNCHER(ReflectPad, float);
DEFINE_KERNEL_LAUNCHER(ReflectPad, double);

DEFINE_KERNEL_LAUNCHER(EdgePad, bool);
DEFINE_KERNEL_LAUNCHER(EdgePad, int8_t);
DEFINE_KERNEL_LAUNCHER(EdgePad, uint8_t);
DEFINE_KERNEL_LAUNCHER(EdgePad, int);
DEFINE_KERNEL_LAUNCHER(EdgePad, int64_t);
DEFINE_KERNEL_LAUNCHER(EdgePad, float16);
DEFINE_KERNEL_LAUNCHER(EdgePad, float);
DEFINE_KERNEL_LAUNCHER(EdgePad, double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_CONST_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
