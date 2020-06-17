#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_ReduceLoss(const int nthreads, const T scale, const T* x, T* y) {
  __shared__ typename BlockReduce<T>::TempStorage storage;
  T val = T(0);
  CUDA_2D_KERNEL_LOOP2(i, nthreads) {
    val += x[i];
  }
  val = BlockReduce<T>(storage).Sum(val);
  if (threadIdx.x == 0) {
    y[0] = val * scale;
  }
}

__global__ void
_ReduceLoss(const int nthreads, const float scale, const half* x, half* y) {
  __shared__ typename BlockReduce<float>::TempStorage storage;
  float val = 0.f;
  CUDA_2D_KERNEL_LOOP2(i, nthreads) {
    val += __half2float(x[i]);
  }
  val = BlockReduce<float>(storage).Sum(val);
  if (threadIdx.x == 0) {
    y[0] = __float2half(val * scale);
  }
}

template <typename T>
__global__ void
_ReduceLossWithMask(const int nthreads, const T* x, const int* mask, T* y) {
  __shared__ union {
    typename BlockReduce<T>::TempStorage loss;
    typename BlockReduce<int>::TempStorage mask;
  } storage;
  T val = T(0);
  int num_valids = 0;
  CUDA_2D_KERNEL_LOOP2(i, nthreads) {
    val += x[i];
    num_valids += mask[i];
  }
  val = BlockReduce<T>(storage.loss).Sum(val);
  num_valids = BlockReduce<int>(storage.mask).Sum(num_valids);
  if (threadIdx.x == 0) {
    y[0] = val / (T)max(1, num_valids);
  }
}

template <>
__global__ void _ReduceLossWithMask<half>(
    const int nthreads,
    const half* x,
    const int* mask,
    half* y) {
  __shared__ union {
    typename BlockReduce<float>::TempStorage loss;
    typename BlockReduce<int>::TempStorage mask;
  } storage;
  float val = 0.f;
  int num_valids = 0;
  CUDA_2D_KERNEL_LOOP2(i, nthreads) {
    val += __half2float(x[i]);
    num_valids += mask[i];
  }
  val = BlockReduce<float>(storage.loss).Sum(val);
  num_valids = BlockReduce<int>(storage.mask).Sum(num_valids);
  if (threadIdx.x == 0) {
    y[0] = __float2half(val / (float)max(1, num_valids));
  }
}

template <typename T>
__global__ void
_ReduceLossGrad(const int nthreads, const T scale, const T* dy, T* dx) {
#if __CUDA_ARCH__ >= 350
  const T val = __ldg(dy) * scale;
#else
  const T val = dy[0] * scale;
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] *= val;
  }
}

__global__ void _ReduceLossGrad(
    const int nthreads,
    const float scale,
    const half* dy,
    half* dx) {
#if __CUDA_ARCH__ >= 350
  const float val = __half2float(__ldg(dy)) * scale;
#else
  const float val = __half2float(dy[0]) * scale;
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] = __float2half(__half2float(dx[i]) * val);
  }
}

__global__ void _ReduceMask(const int num_masks, int* mask) {
  __shared__ typename BlockReduce<int>::TempStorage storage;
  int num_valids = 0;
  CUDA_2D_KERNEL_LOOP2(i, num_masks) {
    num_valids += mask[i];
  }
  num_valids = BlockReduce<int>(storage).Sum(num_valids);
  if (threadIdx.x == 0) mask[0] = max(num_valids, 1);
}

template <typename T>
__global__ void _ReduceLossGradWithMask(
    const int nthreads,
    const T* dy,
    const int* mask,
    T* dx) {
#if __CUDA_ARCH__ >= 350
  const T val = __ldg(dy) / (T)__ldg(mask);
#else
  const T val = dy[0] / (T)mask[0];
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] *= val;
  }
}

template <>
__global__ void _ReduceLossGradWithMask<half>(
    const int nthreads,
    const half* dy,
    const int* mask,
    half* dx) {
#if __CUDA_ARCH__ >= 350
  const float val = __half2float(__ldg(dy)) / (float)__ldg(mask);
#else
  const float val = __half2float(dy[0]) / (float)mask[0];
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] = __float2half(__half2float(dx[i]) * val);
  }
}

template <typename T>
__global__ void _BroadcastLossGrad(
    const int nthreads,
    const int rows,
    const int cols,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] *= __ldg(dy + (i / rows) * cols + (i % cols));
#else
    dx[i] *= dy[(i / rows) * cols + (i % cols)];
#endif
  }
}

template <>
__global__ void _BroadcastLossGrad<half>(
    const int nthreads,
    const int rows,
    const int cols,
    const half* dy,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = __float2half(
        __half2float(dx[i]) *
        __half2float(__ldg(dy + (i / rows) * cols + (i % cols))));
#else
    dx[i] = __float2half(
        __half2float(dx[i]) * __half2float(dy[(i / rows) * cols + (i % cols)]));
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ReduceLoss<float16, CUDAContext>(
    const int count,
    const int num_masks,
    const float normalizer,
    const float16* x,
    const int* mask,
    float16* y,
    CUDAContext* ctx) {
  if (num_masks > 0) {
    _ReduceLossWithMask<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        num_masks,
        reinterpret_cast<const half*>(x),
        mask,
        reinterpret_cast<half*>(y));
  } else {
    _ReduceLoss<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        1.f / std::max(1e-5F, normalizer),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

template <>
void ReduceLossGrad<float16, CUDAContext>(
    const int count,
    const int num_masks,
    const float normalizer,
    const float16* dy,
    const int* mask,
    float16* dx,
    CUDAContext* ctx) {
  if (num_masks > 0) {
    _ReduceMask<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        num_masks, const_cast<int*>(mask));
    _ReduceLossGradWithMask<<<
        CUDA_BLOCKS(count),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        count,
        reinterpret_cast<const half*>(dy),
        mask,
        reinterpret_cast<half*>(dx));
  } else {
    _ReduceLossGrad<<<
        CUDA_BLOCKS(count),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        count,
        1.f / std::max(1e-5F, normalizer),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  }
} // ReduceLossGrad

template <>
void BroadcastLossGrad<float16, CUDAContext>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float16* dy,
    float16* dx,
    CUDAContext* ctx) {
  auto rows = outer_dim * axis_dim, cols = inner_dim;
  auto nthreads = rows * cols;
  _BroadcastLossGrad<<<
      CUDA_BLOCKS(nthreads),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(
      nthreads,
      rows,
      cols,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
} // BroadcastLossGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void ReduceLoss<T, CUDAContext>(                                     \
      const int count,                                                 \
      const int num_masks,                                             \
      const float normalizer,                                          \
      const T* x,                                                      \
      const int* mask,                                                 \
      T* y,                                                            \
      CUDAContext* ctx) {                                              \
    if (num_masks > 0) {                                               \
      _ReduceLossWithMask<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          num_masks, x, mask, y);                                      \
    } else {                                                           \
      _ReduceLoss<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(         \
          count, T(1) / (T)std::max(1e-5F, normalizer), x, y);         \
    }                                                                  \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                           \
  template <>                                                    \
  void ReduceLossGrad<T, CUDAContext>(                           \
      const int count,                                           \
      const int num_masks,                                       \
      const float normalizer,                                    \
      const T* dy,                                               \
      const int* mask,                                           \
      T* dx,                                                     \
      CUDAContext* ctx) {                                        \
    if (num_masks > 0) {                                         \
      _ReduceMask<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
          num_masks, const_cast<int*>(mask));                    \
      _ReduceLossGradWithMask<<<                                 \
          CUDA_BLOCKS(count),                                    \
          CUDA_THREADS,                                          \
          0,                                                     \
          ctx->cuda_stream()>>>(count, dy, mask, dx);            \
    } else {                                                     \
      _ReduceLossGrad<<<                                         \
          CUDA_BLOCKS(count),                                    \
          CUDA_THREADS,                                          \
          0,                                                     \
          ctx->cuda_stream()>>>(                                 \
          count, T(1) / (T)std::max(1e-5F, normalizer), dy, dx); \
    }                                                            \
  }                                                              \
  template <>                                                    \
  void BroadcastLossGrad<T, CUDAContext>(                        \
      const int outer_dim,                                       \
      const int axis_dim,                                        \
      const int inner_dim,                                       \
      const T* dy,                                               \
      T* dx,                                                     \
      CUDAContext* ctx) {                                        \
    auto rows = outer_dim * axis_dim, cols = inner_dim;          \
    auto nthreads = rows * cols;                                 \
    _BroadcastLossGrad<<<                                        \
        CUDA_BLOCKS(nthreads),                                   \
        CUDA_THREADS,                                            \
        0,                                                       \
        ctx->cuda_stream()>>>(nthreads, rows, cols, dy, dx);     \
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
