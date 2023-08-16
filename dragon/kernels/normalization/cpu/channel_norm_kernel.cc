#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename OutputT>
void _ChannelNorm(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const InputT* x,
    const float* mean,
    const float* std,
    OutputT* y) {
  const auto N = math::utils::Prod(num_dims, y_dims);
  vec64_t idx(num_dims, 0);
  for (int yi = 0; yi < N; ++yi) {
    int64_t xi = 0, wi;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += idx[d] * x_strides[d];
      if (d == axis) wi = idx[d];
    }
    const float val = convert::To<float>(x[xi]);
    y[yi] = convert::To<OutputT>((val - mean[wi]) / std[wi]);
    math::utils::IncreaseIndexInDims(num_dims, y_dims, idx.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(InputT, OutputT)                       \
  template <>                                                         \
  void ChannelNorm<InputT, OutputT, CPUContext>(                      \
      const int axis,                                                 \
      const int num_dims,                                             \
      const int64_t* x_strides,                                       \
      const int64_t* y_dims,                                          \
      const InputT* x,                                                \
      const float* mean,                                              \
      const float* std,                                               \
      OutputT* y,                                                     \
      CPUContext* ctx) {                                              \
    _ChannelNorm(axis, num_dims, x_strides, y_dims, x, mean, std, y); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t, float16);
DEFINE_KERNEL_LAUNCHER(uint8_t, bfloat16);
DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(uint8_t, double);
DEFINE_KERNEL_LAUNCHER(int8_t, float16);
DEFINE_KERNEL_LAUNCHER(int8_t, bfloat16);
DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(int8_t, double);
DEFINE_KERNEL_LAUNCHER(int, float16);
DEFINE_KERNEL_LAUNCHER(int, bfloat16);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int, double);
DEFINE_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_KERNEL_LAUNCHER(int64_t, bfloat16);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float16, float16);
DEFINE_KERNEL_LAUNCHER(float16, bfloat16);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float16, double);
DEFINE_KERNEL_LAUNCHER(bfloat16, float16);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, bfloat16);
DEFINE_KERNEL_LAUNCHER(bfloat16, double);
DEFINE_KERNEL_LAUNCHER(float, float16);
DEFINE_KERNEL_LAUNCHER(float, bfloat16);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(float, double);
DEFINE_KERNEL_LAUNCHER(double, float16);
DEFINE_KERNEL_LAUNCHER(double, bfloat16);
DEFINE_KERNEL_LAUNCHER(double, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
