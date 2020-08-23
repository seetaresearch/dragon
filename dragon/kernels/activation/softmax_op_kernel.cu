#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/cast.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Softmax(
    const int rows,
    const int cols,
    const int inner_dim,
    const T lowest,
    const T* x,
    T* y) {
  __shared__ T block_val;
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    const int c = (i / inner_dim) * cols * inner_dim + (i % inner_dim);

    T val = lowest;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
#if __CUDA_ARCH__ >= 350
      val = max(val, __ldg(x + yi));
#else
      val = max(val, x[yi]);
#endif
    }
    val = BlockReduce<T>(storage).Reduce(val, cub::Max());
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
#if __CUDA_ARCH__ >= 350
      y[yi] = exp(__ldg(x + yi) - block_val);
#else
      y[yi] = exp(x[yi] - block_val);
#endif
    }

    val = T(0);
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      val += y[yi];
    }
    val = BlockReduce<T>(storage).Sum(val);
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      y[yi] /= block_val;
    }
  }
}

template <>
__global__ void _Softmax<half>(
    const int rows,
    const int cols,
    const int inner_dim,
    const half lowest,
    const half* x,
    half* y) {
#if __CUDA_ARCH__ >= 530
  __shared__ float block_val;
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    const int c = (i / inner_dim) * cols * inner_dim + (i % inner_dim);

    float val = __half2float(lowest);
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      val = max(val, __half2float(__ldg(x + yi)));
    }
    val = BlockReduce<float>(storage).Reduce(val, cub::Max());
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      y[yi] = __float2half(exp(__half2float(__ldg(x + yi)) - block_val));
    }

    val = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      val += __half2float(y[yi]);
    }
    val = BlockReduce<float>(storage).Sum(val);
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      y[yi] = __float2half(__half2float(y[yi]) / block_val);
    }
  }
#endif
}

template <typename T>
__global__ void _SoftmaxGrad(
    const int rows,
    const int cols,
    const int inner_dim,
    const T* dy,
    const T* y,
    T* dx) {
  __shared__ T block_val;
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    const int c = (i / inner_dim) * cols * inner_dim + (i % inner_dim);

    T val = T(0);
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
#if __CUDA_ARCH__ >= 350
      val += __ldg(dy + yi) * __ldg(y + yi);
#else
      val += dy[yi] * y[yi];
#endif
    }
    val = BlockReduce<T>(storage).Sum(val);
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
#if __CUDA_ARCH__ >= 350
      dx[yi] = (__ldg(dy + yi) - block_val) * __ldg(y + yi);
#else
      dx[yi] = (dy[yi] - block_val) * y[yi];
#endif
    }
  }
}

template <>
__global__ void _SoftmaxGrad<half>(
    const int rows,
    const int cols,
    const int inner_dim,
    const half* dy,
    const half* y,
    half* dx) {
#if __CUDA_ARCH__ >= 530
  __shared__ float block_val;
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    const int c = (i / inner_dim) * cols * inner_dim + (i % inner_dim);

    float val = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      val += __half2float(__ldg(dy + yi)) * __half2float(__ldg(y + yi));
    }
    val = BlockReduce<float>(storage).Sum(val);
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int yi = c + j * inner_dim;
      dx[yi] = __float2half(
          (__half2float(__ldg(dy + yi)) - block_val) *
          __half2float(__ldg(y + yi)));
    }
  }
#endif
} // SoftmaxGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Softmax<float16, CUDAContext>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  auto rows = outer_dim * inner_dim, cols = axis_dim;
  _Softmax<<<CUDA_2D_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      rows,
      cols,
      inner_dim,
      cast::to<half>(std::numeric_limits<float>::lowest()),
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y));
}

template <>
void SoftmaxGrad<float16, CUDAContext>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  auto rows = outer_dim * inner_dim, cols = axis_dim;
  _SoftmaxGrad<<<CUDA_2D_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      rows,
      cols,
      inner_dim,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<const half*>(y),
      reinterpret_cast<half*>(dx));
}

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void Softmax<T, CUDAContext>(                                              \
      const int outer_dim,                                                   \
      const int axis_dim,                                                    \
      const int inner_dim,                                                   \
      const T* x,                                                            \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    auto rows = outer_dim * inner_dim, cols = axis_dim;                      \
    _Softmax<<<CUDA_2D_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        rows, cols, inner_dim, std::numeric_limits<T>::lowest(), x, y);      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                           \
  template <>                                                    \
  void SoftmaxGrad<T, CUDAContext>(                              \
      const int outer_dim,                                       \
      const int axis_dim,                                        \
      const int inner_dim,                                       \
      const T* dy,                                               \
      const T* y,                                                \
      T* dx,                                                     \
      CUDAContext* ctx) {                                        \
    auto rows = outer_dim * inner_dim, cols = axis_dim;          \
    _SoftmaxGrad<<<                                              \
        CUDA_2D_BLOCKS(rows),                                    \
        CUDA_THREADS,                                            \
        0,                                                       \
        ctx->cuda_stream()>>>(rows, cols, inner_dim, dy, y, dx); \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
