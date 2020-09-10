#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _L1Normalize(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const T scale,
    const T eps,
    const T* x,
    T* y) {
  __shared__ T norm;
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    T sum = T(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      sum += abs(x[offset + j * inner_dim]);
    }
    sum = BlockReduce<T>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sum * scale, eps);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      y[idx] = x[idx] / norm;
    }
  }
}

__global__ void _L1Normalize(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const half* x,
    half* y) {
  __shared__ float norm;
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    float sum = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      sum += abs(__half2float(x[offset + j * inner_dim]));
    }
    sum = BlockReduce<float>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sum * scale, eps);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      y[idx] = __float2half(__half2float(x[idx]) / norm);
    }
  }
}

template <typename T>
__global__ void _L2Normalize(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const T scale,
    const T eps,
    const T* x,
    T* y) {
  __shared__ T norm;
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    T sum = T(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      sum += utils::math::Square(x[offset + j * inner_dim]);
    }
    sum = BlockReduce<T>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sqrt(sum * scale), eps);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      y[idx] = x[idx] / norm;
    }
  }
}

__global__ void _L2Normalize(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const half* x,
    half* y) {
  __shared__ float norm;
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    float sum = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      sum += utils::math::Square(__half2float(x[offset + j * inner_dim]));
    }
    sum = BlockReduce<float>(storage).Sum(sum);
    if (threadIdx.x == 0) {
      norm = max(sqrt(sum * scale), eps);
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      y[idx] = __float2half(__half2float(x[idx]) / norm);
    }
  }
}

template <typename T>
__global__ void _L1NormalizeGrad(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const T scale,
    const T eps,
    const T* dy,
    const T* x,
    T* dx) {
  __shared__ T norm, norm2, sum;
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    T val1 = T(0), val2 = T(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      val1 += abs(x[idx]);
      val2 += dy[idx] * x[idx];
    }
    val1 = BlockReduce<T>(storage).Sum(val1);
    val2 = BlockReduce<T>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(val1 * scale, eps);
      norm2 = pow(norm, 2);
      sum = val2 * scale;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      dx[idx] = (dy[idx] / norm) - ((utils::math::Sign(x[idx]) / norm2) * sum);
    }
  }
}

__global__ void _L1NormalizeGrad(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const half* dy,
    const half* x,
    half* dx) {
  __shared__ float norm, norm2, sum;
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    float val1 = 0.f, val2 = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      val1 += abs(__half2float(x[idx]));
      val2 += __half2float(dy[idx]) * __half2float(x[idx]);
    }
    val1 = BlockReduce<float>(storage).Sum(val1);
    val2 = BlockReduce<float>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(val1 * scale, eps);
      norm2 = pow(norm, 2);
      sum = val2 * scale;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      dx[idx] = __float2half(
          (__half2float(dy[idx]) / norm) -
          ((utils::math::Sign(__half2float(x[idx])) / norm2) * sum));
    }
  }
}

template <typename T>
__global__ void _L2NormalizeGrad(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const T scale,
    const T eps,
    const T* dy,
    const T* x,
    T* dx) {
  __shared__ T norm, norm3, sum;
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    T val1 = T(0), val2 = T(0);
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      val1 += utils::math::Square(x[idx]);
      val2 += dy[idx] * x[idx];
    }
    val1 = BlockReduce<T>(storage).Sum(val1);
    val2 = BlockReduce<T>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(sqrt(val1 * scale), eps);
      norm3 = pow(norm, 3);
      sum = val2 * scale;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      dx[idx] = (dy[idx] / norm) - ((x[idx] / norm3) * sum);
    }
  }
}

__global__ void _L2NormalizeGrad(
    const int nblocks,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const half* dy,
    const half* x,
    half* dx) {
  __shared__ float norm, norm3, sum;
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, nblocks) {
    auto offset = (i / inner_dim) * reduce_dim * inner_dim + (i % inner_dim);
    float val1 = 0.f, val2 = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      val1 += utils::math::Square(__half2float(x[idx]));
      val2 += __half2float(dy[idx]) * __half2float(x[idx]);
    }
    val1 = BlockReduce<float>(storage).Sum(val1);
    val2 = BlockReduce<float>(storage).Sum(val2);
    if (threadIdx.x == 0) {
      norm = max(sqrt(val1 * scale), eps);
      norm3 = pow(norm, 3);
      sum = val2 * scale;
    }
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, reduce_dim) {
      auto idx = offset + j * inner_dim;
      dx[idx] = __float2half(
          (__half2float(dy[idx]) / norm) -
          ((__half2float(x[idx]) / norm3) * sum));
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                                        \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int outer_dim,                                                     \
      const int reduce_dim,                                                    \
      const int inner_dim,                                                     \
      const float scale,                                                       \
      const float eps,                                                         \
      const float16* x,                                                        \
      float16* y,                                                              \
      CUDAContext* ctx) {                                                      \
    const auto nblocks = outer_dim * inner_dim;                                \
    _##name<<<CUDA_2D_BLOCKS(nblocks), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nblocks,                                                               \
        reduce_dim,                                                            \
        inner_dim,                                                             \
        scale,                                                                 \
        eps,                                                                   \
        reinterpret_cast<const half*>(x),                                      \
        reinterpret_cast<half*>(y));                                           \
  }

DEFINE_KERNEL_LAUNCHER(L1Normalize, float16);
DEFINE_KERNEL_LAUNCHER(L2Normalize, float16);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)                                        \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int outer_dim,                                                     \
      const int reduce_dim,                                                    \
      const int inner_dim,                                                     \
      const float scale,                                                       \
      const float eps,                                                         \
      const T* x,                                                              \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    const auto nblocks = outer_dim * inner_dim;                                \
    _##name<<<CUDA_2D_BLOCKS(nblocks), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nblocks, reduce_dim, inner_dim, (T)scale, (T)eps, x, y);               \
  }

DEFINE_KERNEL_LAUNCHER(L1Normalize, float);
DEFINE_KERNEL_LAUNCHER(L1Normalize, double);
DEFINE_KERNEL_LAUNCHER(L2Normalize, float);
DEFINE_KERNEL_LAUNCHER(L2Normalize, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                   \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int outer_dim,                                                     \
      const int reduce_dim,                                                    \
      const int inner_dim,                                                     \
      const float scale,                                                       \
      const float eps,                                                         \
      const float16* dy,                                                       \
      const float16* x,                                                        \
      T* dx,                                                                   \
      CUDAContext* ctx) {                                                      \
    const auto nblocks = outer_dim * inner_dim;                                \
    _##name<<<CUDA_2D_BLOCKS(nblocks), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nblocks,                                                               \
        reduce_dim,                                                            \
        inner_dim,                                                             \
        scale,                                                                 \
        eps,                                                                   \
        reinterpret_cast<const half*>(dy),                                     \
        reinterpret_cast<const half*>(x),                                      \
        reinterpret_cast<half*>(dx));                                          \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, float16);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                   \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int outer_dim,                                                     \
      const int reduce_dim,                                                    \
      const int inner_dim,                                                     \
      const float scale,                                                       \
      const float eps,                                                         \
      const T* dy,                                                             \
      const T* x,                                                              \
      T* dx,                                                                   \
      CUDAContext* ctx) {                                                      \
    const auto nblocks = outer_dim * inner_dim;                                \
    _##name<<<CUDA_2D_BLOCKS(nblocks), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nblocks, reduce_dim, inner_dim, (T)scale, (T)eps, dy, x, dx);          \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormalizeGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormalizeGrad, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
