#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Softmax(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    T* y) {
  int row_offset, col_offset, yi;
  auto x_stride = axis_dim * inner_dim;
  for (int i = 0; i < outer_dim; ++i) {
    row_offset = i * axis_dim * inner_dim;
    for (int j = 0; j < inner_dim; ++j) {
      col_offset = row_offset + j;
      T val = x[col_offset];
      for (int k = 1; k < axis_dim; ++k) {
        yi = col_offset + k * inner_dim;
        val = std::max(val, x[yi]);
      }
      for (int k = 0; k < axis_dim; ++k) {
        yi = col_offset + k * inner_dim;
        y[yi] = std::exp(x[yi] - val);
      }
      val = y[col_offset];
      for (int k = 1; k < axis_dim; ++k) {
        yi = col_offset + k * inner_dim;
        val += y[yi];
      }
      for (int k = 0; k < axis_dim; ++k) {
        yi = col_offset + k * inner_dim;
        y[yi] /= val;
      }
    }
  }
}

template <>
void _Softmax<float16>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _SoftmaxGrad(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const T* dy,
    const T* y,
    T* dx) {
  int row_offset, col_offset, yi;
  auto x_stride = axis_dim * inner_dim;
  for (int i = 0; i < outer_dim; ++i) {
    row_offset = i * axis_dim * inner_dim;
    for (int j = 0; j < inner_dim; ++j) {
      col_offset = row_offset + j;
      T val = dy[col_offset] * y[col_offset];
      for (int k = 1; k < axis_dim; ++k) {
        yi = col_offset + k * inner_dim;
        val += dy[yi] * y[yi];
      }
      for (int k = 0; k < axis_dim; ++k) {
        yi = col_offset + k * inner_dim;
        dx[yi] = (dy[yi] - val) * y[yi];
      }
    }
  }
}

template <>
void _SoftmaxGrad<float16>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // SoftmaxGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                   \
  template <>                                       \
  void Softmax<T, CPUContext>(                      \
      const int outer_dim,                          \
      const int inner_dim,                          \
      const int axis_dim,                           \
      const T* x,                                   \
      T* y,                                         \
      CPUContext* ctx) {                            \
    _Softmax(outer_dim, inner_dim, axis_dim, x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                       \
  template <>                                                \
  void SoftmaxGrad<T, CPUContext>(                           \
      const int outer_dim,                                   \
      const int inner_dim,                                   \
      const int axis_dim,                                    \
      const T* dy,                                           \
      const T* y,                                            \
      T* dx,                                                 \
      CPUContext* ctx) {                                     \
    _SoftmaxGrad(outer_dim, inner_dim, axis_dim, dy, y, dx); \
  }

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
