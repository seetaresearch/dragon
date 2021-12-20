#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename OutputT>
void _ChannelNormalize(
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
  int64_t xi, wi;
  for (int yi = 0; yi < N; ++yi) {
    xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += idx[d] * x_strides[d];
      if (d == axis) wi = idx[d];
    }
    y[yi] =
        convert::To<OutputT>((convert::To<float>(x[xi]) - mean[wi]) / std[wi]);
    math::utils::IncreaseIndexInDims(num_dims, y_dims, idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(InputT, OutputT)                            \
  template <>                                                              \
  void ChannelNormalize<InputT, OutputT, CPUContext>(                      \
      const int axis,                                                      \
      const int num_dims,                                                  \
      const int64_t* x_strides,                                            \
      const int64_t* y_dims,                                               \
      const InputT* x,                                                     \
      const float* mean,                                                   \
      const float* std,                                                    \
      OutputT* y,                                                          \
      CPUContext* ctx) {                                                   \
    _ChannelNormalize(axis, num_dims, x_strides, y_dims, x, mean, std, y); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t, float16);
DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(uint8_t, double);
DEFINE_KERNEL_LAUNCHER(int8_t, float16);
DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(int8_t, double);
DEFINE_KERNEL_LAUNCHER(int, float16);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int, double);
DEFINE_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float16, float16);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float16, double);
DEFINE_KERNEL_LAUNCHER(float, float16);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(float, double);
DEFINE_KERNEL_LAUNCHER(double, float16);
DEFINE_KERNEL_LAUNCHER(double, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
