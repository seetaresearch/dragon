#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename Tx, typename Ty, int D>
__global__ void _ChannelNormalize(
    const int nthreads,
    const int axis,
    const int num_dims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const Tx* x,
    const float* mean,
    const float* std,
    Ty* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, wi, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
      if (d == axis) wi = r;
    }
#if __CUDA_ARCH__ >= 350
    y[yi] = ((Ty)x[xi] - (Ty)__ldg(mean + wi)) / (Ty)__ldg(std + wi);
#else
    y[yi] = ((Ty)x[xi] - (Ty)mean[wi]) / (Ty)std[wi];
#endif
  }
}

template <typename T, int D>
__global__ void _ChannelNormalizeHalf(
    const int nthreads,
    const int axis,
    const int num_dims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const T* x,
    const float* mean,
    const float* std,
    half* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, wi, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
      if (d == axis) wi = r;
    }
#if __CUDA_ARCH__ >= 350
    y[yi] = __float2half(((float)x[xi] - __ldg(mean + wi)) / __ldg(std + wi));
#else
    y[yi] = __float2half(((float)x[xi] - mean[wi]) / std[wi]);
#endif
  }
}

template <typename T, int D>
__global__ void _ChannelNormalizeHalf(
    const int nthreads,
    const int axis,
    const int num_dims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const half* x,
    const float* mean,
    const float* std,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, wi, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
      if (d == axis) wi = r;
    }
#if __CUDA_ARCH__ >= 350
    y[yi] = (T)((__half2float(x[xi]) - __ldg(mean + wi)) / __ldg(std + wi));
#else
    y[yi] = (T)((__half2float(x[xi]) - mean[wi]) / std[wi]);
#endif
  }
}

template <int D>
__global__ void _ChannelNormalizeHalfAndHalf(
    const int nthreads,
    const int axis,
    const int num_dims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const half* x,
    const float* mean,
    const float* std,
    half* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, wi, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
      if (d == axis) wi = r;
    }
#if __CUDA_ARCH__ >= 350
    y[yi] = __float2half(
        ((__half2float(x[xi]) - __ldg(mean + wi)) / __ldg(std + wi)));
#else
    y[yi] = __float2half(((__half2float(x[xi]) - mean[wi]) / std[wi]));
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ChannelNormalize<float16, float16, CUDAContext>(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const float16* x,
    const float* mean,
    const float* std,
    float16* y,
    CUDAContext* ctx) {
  CUDA_TENSOR_DIMS_CHECK(num_dims);
  SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims;
  const auto nthreads =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  for (int i = 0; i < num_dims; ++i) {
    X_strides.data[i] = x_strides[i];
    Y_dims.data[i] = y_dims[i];
  }
  _ChannelNormalizeHalfAndHalf<<<
      CUDA_BLOCKS(nthreads),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(
      nthreads,
      axis,
      num_dims,
      X_strides,
      Y_dims,
      reinterpret_cast<const half*>(x),
      mean,
      std,
      reinterpret_cast<half*>(y));
}

#define DEFINE_KERNEL_LAUNCHER(Tx, Ty)                                 \
  template <>                                                          \
  void ChannelNormalize<Tx, Ty, CUDAContext>(                          \
      const int axis,                                                  \
      const int num_dims,                                              \
      const int64_t* x_strides,                                        \
      const int64_t* y_dims,                                           \
      const Tx* x,                                                     \
      const float* mean,                                               \
      const float* std,                                                \
      Ty* y,                                                           \
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

#define DEFINE_FP16_KERNEL_LAUNCHER(T)                             \
  template <>                                                      \
  void ChannelNormalize<float16, T, CUDAContext>(                  \
      const int axis,                                              \
      const int num_dims,                                          \
      const int64_t* x_strides,                                    \
      const int64_t* y_dims,                                       \
      const float16* x,                                            \
      const float* mean,                                           \
      const float* std,                                            \
      T* y,                                                        \
      CUDAContext* ctx) {                                          \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                              \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims;      \
    const auto nthreads = std::accumulate(                         \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>()); \
    for (int i = 0; i < num_dims; ++i) {                           \
      X_strides.data[i] = x_strides[i];                            \
      Y_dims.data[i] = y_dims[i];                                  \
    }                                                              \
    _ChannelNormalizeHalf<<<                                       \
        CUDA_BLOCKS(nthreads),                                     \
        CUDA_THREADS,                                              \
        0,                                                         \
        ctx->cuda_stream()>>>(                                     \
        nthreads,                                                  \
        axis,                                                      \
        num_dims,                                                  \
        X_strides,                                                 \
        Y_dims,                                                    \
        reinterpret_cast<const half*>(x),                          \
        mean,                                                      \
        std,                                                       \
        y);                                                        \
  }                                                                \
  template <>                                                      \
  void ChannelNormalize<T, float16, CUDAContext>(                  \
      const int axis,                                              \
      const int num_dims,                                          \
      const int64_t* x_strides,                                    \
      const int64_t* y_dims,                                       \
      const T* x,                                                  \
      const float* mean,                                           \
      const float* std,                                            \
      float16* y,                                                  \
      CUDAContext* ctx) {                                          \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                              \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims;      \
    const auto nthreads = std::accumulate(                         \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>()); \
    for (int i = 0; i < num_dims; ++i) {                           \
      X_strides.data[i] = x_strides[i];                            \
      Y_dims.data[i] = y_dims[i];                                  \
    }                                                              \
    _ChannelNormalizeHalf<<<                                       \
        CUDA_BLOCKS(nthreads),                                     \
        CUDA_THREADS,                                              \
        0,                                                         \
        ctx->cuda_stream()>>>(                                     \
        nthreads,                                                  \
        axis,                                                      \
        num_dims,                                                  \
        X_strides,                                                 \
        Y_dims,                                                    \
        x,                                                         \
        mean,                                                      \
        std,                                                       \
        reinterpret_cast<half*>(y));                               \
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

#endif // USE_CUDA
