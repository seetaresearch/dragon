#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, int D>
__global__ void _Tile(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += (r % x_dims.data[d]) * x_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T>
__global__ void _TileGrad(
    const int nthreads,
    const int x_cols,
    const int y_cols,
    const int multiple,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int i = xi / x_cols;
    const int j = xi % x_cols;
    const T* offset_dy = dy + i * y_cols + j;
    T val = (*offset_dy);
    offset_dy += x_cols;
    for (int k = 1; k < multiple; ++k) {
      val += (*offset_dy);
      offset_dy += x_cols;
    }
    dx[xi] = val;
  }
}

template <>
__global__ void _TileGrad<half>(
    const int nthreads,
    const int x_cols,
    const int y_cols,
    const int multiple,
    const half* dy,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int i = xi / x_cols;
    const int j = xi % x_cols;
    const half* offset_dy = dy + i * y_cols + j;
    float val = __half2float(*offset_dy);
    offset_dy += x_cols;
    for (int k = 1; k < multiple; ++k) {
      val += __half2float(*offset_dy);
      offset_dy += x_cols;
    }
    dx[xi] = __float2half(val);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void TileGrad<float16, CUDAContext>(
    const int rows,
    const int cols,
    const int multiple,
    const float16* dy,
    float16* dx,
    CUDAContext* ctx) {
  const int nthreads = rows * cols;
  _TileGrad<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      nthreads,
      cols,
      cols * multiple,
      multiple,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
} // TileGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void Tile<T, CUDAContext>(                                               \
      const int num_dims,                                                  \
      const int64_t* x_dims,                                               \
      const int64_t* x_strides,                                            \
      const int64_t* y_dims,                                               \
      const T* x,                                                          \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                      \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, X_strides, Y_dims;      \
    const auto nthreads = std::accumulate(                                 \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());         \
    for (int i = 0; i < num_dims; ++i) {                                   \
      X_dims.data[i] = x_dims[i];                                          \
      X_strides.data[i] = x_strides[i];                                    \
      Y_dims.data[i] = y_dims[i];                                          \
    }                                                                      \
    _Tile<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, num_dims, X_dims, X_strides, Y_dims, x, y);              \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                         \
  template <>                                                                  \
  void TileGrad<T, CUDAContext>(                                               \
      const int rows,                                                          \
      const int cols,                                                          \
      const int multiple,                                                      \
      const T* dy,                                                             \
      T* dx,                                                                   \
      CUDAContext* ctx) {                                                      \
    const int nthreads = rows * cols;                                          \
    _TileGrad<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, cols, cols * multiple, multiple, dy, dx);                    \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
