#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename Tx, typename Ty>
void _ChannelNormalize(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const Tx* x,
    const float* mean,
    const float* std,
    Ty* y) {
  const auto count =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t idx(num_dims, 0);
  int64_t xi, wi;
  for (int yi = 0; yi < count; ++yi) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += idx[d] * x_strides[d];
      if (d == axis) wi = idx[d];
    }
    y[yi] = ((Ty)x[xi] - (Ty)mean[wi]) / (Ty)std[wi];
    utils::math::IncreaseIndexInDims(num_dims, y_dims, idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ChannelNormalize<float16, float16, CPUContext>(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const float16* x,
    const float* mean,
    const float* std,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(Tx, Ty)                                     \
  template <>                                                              \
  void ChannelNormalize<Tx, Ty, CPUContext>(                               \
      const int axis,                                                      \
      const int num_dims,                                                  \
      const int64_t* x_strides,                                            \
      const int64_t* y_dims,                                               \
      const Tx* x,                                                         \
      const float* mean,                                                   \
      const float* std,                                                    \
      Ty* y,                                                               \
      CPUContext* ctx) {                                                   \
    _ChannelNormalize(axis, num_dims, x_strides, y_dims, x, mean, std, y); \
  }

#define DEFINE_FP16_KERNEL_LAUNCHER(T)           \
  template <>                                    \
  void ChannelNormalize<float16, T, CPUContext>( \
      const int axis,                            \
      const int num_dims,                        \
      const int64_t* x_strides,                  \
      const int64_t* y_dims,                     \
      const float16* x,                          \
      const float* mean,                         \
      const float* std,                          \
      T* y,                                      \
      CPUContext* ctx) {                         \
    CPU_FP16_NOT_SUPPORTED;                      \
  }                                              \
  template <>                                    \
  void ChannelNormalize<T, float16, CPUContext>( \
      const int axis,                            \
      const int num_dims,                        \
      const int64_t* x_strides,                  \
      const int64_t* y_dims,                     \
      const T* x,                                \
      const float* mean,                         \
      const float* std,                          \
      float16* y,                                \
      CPUContext* ctx) {                         \
    CPU_FP16_NOT_SUPPORTED;                      \
  }

DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(int8_t, double);
DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(uint8_t, double);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int, double);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(float, double);
DEFINE_KERNEL_LAUNCHER(double, float);
DEFINE_KERNEL_LAUNCHER(double, double);

DEFINE_FP16_KERNEL_LAUNCHER(int8_t);
DEFINE_FP16_KERNEL_LAUNCHER(uint8_t);
DEFINE_FP16_KERNEL_LAUNCHER(int);
DEFINE_FP16_KERNEL_LAUNCHER(int64_t);
DEFINE_FP16_KERNEL_LAUNCHER(float);
DEFINE_FP16_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_FP16_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
