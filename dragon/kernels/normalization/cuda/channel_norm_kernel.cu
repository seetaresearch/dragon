#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename OutputT, int D>
__global__ void _ChannelNorm(
    const int N,
    const int axis,
    const int num_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const InputT* x,
    const float* mean,
    const float* std,
    OutputT* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, wi, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += r * X_strides.data[d];
      if (d == axis) wi = r;
    }
    y[yi] = (convert::To<float>(x[xi]) - __ldg(mean + wi)) / __ldg(std + wi);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(InputT, OutputT)                            \
  template <>                                                              \
  void ChannelNorm<InputT, OutputT, CUDAContext>(                          \
      const int axis,                                                      \
      const int num_dims,                                                  \
      const int64_t* x_strides,                                            \
      const int64_t* y_dims,                                               \
      const InputT* x,                                                     \
      const float* mean,                                                   \
      const float* std,                                                    \
      OutputT* y,                                                          \
      CUDAContext* ctx) {                                                  \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                      \
    const auto N = math::utils::Prod(num_dims, y_dims);                    \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims;              \
    for (int i = 0; i < num_dims; ++i) {                                   \
      X_strides.data[i] = x_strides[i], Y_dims.data[i] = y_dims[i];        \
    }                                                                      \
    _ChannelNorm<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                                 \
        axis,                                                              \
        num_dims,                                                          \
        X_strides,                                                         \
        Y_dims,                                                            \
        reinterpret_cast<const math::Traits<InputT>::scalar_type*>(x),     \
        mean,                                                              \
        std,                                                               \
        reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));         \
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
