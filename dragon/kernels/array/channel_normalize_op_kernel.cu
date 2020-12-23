#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#else
#define LDG(x, i) x[i]
#endif

template <typename InputT, typename OutputT, int D>
__global__ void _ChannelNormalize(
    const int nthreads,
    const int axis,
    const int num_dims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const InputT* x,
    const float* mean,
    const float* std,
    OutputT* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, wi, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
      if (d == axis) wi = r;
    }
    y[yi] = convert::To<OutputT>(
        (convert::To<float>(x[xi]) - LDG(mean, wi)) / LDG(std, wi));
  }
}

#undef LDG

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(InputT, OutputT)                        \
  template <>                                                          \
  void ChannelNormalize<InputT, OutputT, CUDAContext>(                 \
      const int axis,                                                  \
      const int num_dims,                                              \
      const int64_t* x_strides,                                        \
      const int64_t* y_dims,                                           \
      const InputT* x,                                                 \
      const float* mean,                                               \
      const float* std,                                                \
      OutputT* y,                                                      \
      CUDAContext* ctx) {                                              \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                  \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims;          \
    const auto nthreads = std::accumulate(                             \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());     \
    for (int i = 0; i < num_dims; ++i) {                               \
      X_strides.data[i] = x_strides[i];                                \
      Y_dims.data[i] = y_dims[i];                                      \
    }                                                                  \
    _ChannelNormalize<<<                                               \
        CUDA_BLOCKS(nthreads),                                         \
        CUDA_THREADS,                                                  \
        0,                                                             \
        ctx->cuda_stream()>>>(                                         \
        nthreads, axis, num_dims, X_strides, Y_dims, x, mean, std, y); \
  }

DEFINE_KERNEL_LAUNCHER(int8_t, float16);
DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(int8_t, double);
DEFINE_KERNEL_LAUNCHER(uint8_t, float16);
DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(uint8_t, double);
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

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
