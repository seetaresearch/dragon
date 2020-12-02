#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, int D>
__global__ void _ReduceSumGrad(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> y_dims,
    const SimpleArray<int, D> y_strides,
    const float scale,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    int yi = 0, tmp = xi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(x_dims.data[d], tmp, &tmp, &r);
      yi += (r % y_dims.data[d]) * y_strides.data[d];
    }
#if __CUDA_ARCH__ >= 350
    dx[xi] = __ldg(dy + yi) * scale;
#else
    dx[xi] = dy[yi] * scale;
#endif
  }
}

template <int D>
__global__ void _ReduceSumGrad(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> y_dims,
    const SimpleArray<int, D> y_strides,
    const float scale,
    const half* dy,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    int yi = 0, tmp = xi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(x_dims.data[d], tmp, &tmp, &r);
      yi += (r % y_dims.data[d]) * y_strides.data[d];
    }
#if __CUDA_ARCH__ >= 350
    dx[xi] = __float2half(__half2float(__ldg(dy + yi)) * scale);
#else
    dx[xi] = __float2half(__half2float(dy[yi]) * scale);
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ReduceSumGrad<float16, CUDAContext>(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_dims,
    const int64_t* y_strides,
    const float scale,
    const float16* dy,
    float16* dx,
    CUDAContext* ctx) {
  CUDA_TENSOR_DIMS_CHECK(num_dims);
  SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, Y_dims, Y_strides;
  const auto nthreads =
      std::accumulate(x_dims, x_dims + num_dims, 1, std::multiplies<int64_t>());
  for (int i = 0; i < num_dims; ++i) {
    X_dims.data[i] = x_dims[i];
    Y_dims.data[i] = y_dims[i];
    Y_strides.data[i] = y_strides[i];
  }
  _ReduceSumGrad<<<
      CUDA_BLOCKS(nthreads),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(
      nthreads,
      num_dims,
      X_dims,
      Y_dims,
      Y_strides,
      scale,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
} // ReduceSumGrad

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                 \
  template <>                                                          \
  void ReduceSumGrad<T, CUDAContext>(                                  \
      const int num_dims,                                              \
      const int64_t* x_dims,                                           \
      const int64_t* y_dims,                                           \
      const int64_t* y_strides,                                        \
      const float scale,                                               \
      const T* dy,                                                     \
      T* dx,                                                           \
      CUDAContext* ctx) {                                              \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                  \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, Y_dims, Y_strides;  \
    const auto nthreads = std::accumulate(                             \
        x_dims, x_dims + num_dims, 1, std::multiplies<int64_t>());     \
    for (int i = 0; i < num_dims; ++i) {                               \
      X_dims.data[i] = x_dims[i];                                      \
      Y_dims.data[i] = y_dims[i];                                      \
      Y_strides.data[i] = y_strides[i];                                \
    }                                                                  \
    _ReduceSumGrad<<<                                                  \
        CUDA_BLOCKS(nthreads),                                         \
        CUDA_THREADS,                                                  \
        0,                                                             \
        ctx->cuda_stream()>>>(                                         \
        nthreads, num_dims, X_dims, Y_dims, Y_strides, scale, dy, dx); \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
