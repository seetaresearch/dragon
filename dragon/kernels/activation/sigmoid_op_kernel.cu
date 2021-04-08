#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Sigmoid(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = T(1) / (T(1) + exp(-x[i]));
  }
}

template <>
__global__ void _Sigmoid<half>(const int N, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(1.f / (1.f + exp(-__half2float(x[i]))));
  }
}

template <>
__global__ void _Sigmoid<half2>(const int N, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(x[i]);
    y[i] =
        __floats2half2_rn(1.f / (1.f + exp(-val.x)), 1.f / (1.f + exp(-val.y)));
  }
}

template <typename T>
__global__ void _SigmoidGrad(const int N, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * __ldg(y + i) * (1 - __ldg(y + i));
  }
}

template <>
__global__ void
_SigmoidGrad<half>(const int N, const half* dy, const half* y, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] = __float2half(__half2float(dy[i]) * val * (1.f - val));
  }
} // SigmoidGrad

template <>
__global__ void
_SigmoidGrad<half2>(const int N, const half2* dy, const half2* y, half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(
        grad.x * val.x * (1.f - val.x), grad.y * val.y * (1.f - val.y));
  }
} // SigmoidGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Sigmoid<float16, CUDAContext>(
    const int N,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _Sigmoid<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1, reinterpret_cast<const half2*>(x), reinterpret_cast<half2*>(y));
  } else {
    _Sigmoid<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
}

template <>
void SigmoidGrad<float16, CUDAContext>(
    const int N,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _SigmoidGrad<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _SigmoidGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // SigmoidGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void Sigmoid<T, CUDAContext>(                                        \
      const int N, const T* x, T* y, CUDAContext* ctx) {               \
    _Sigmoid<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, x, y);                                                      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                     \
  template <>                                                              \
  void SigmoidGrad<T, CUDAContext>(                                        \
      const int N, const T* dy, const T* y, T* dx, CUDAContext* ctx) {     \
    _SigmoidGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, dy, y, dx);                                                     \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
