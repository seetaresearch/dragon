#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/conversions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_Selu(const int N, const T scale, const T gamma, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > 0 ? gamma * __ldg(x + i)
                            : scale * (exp(__ldg(x + i)) - 1);
  }
}

__global__ void _Selu(
    const int N,
    const float scale,
    const float gamma,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(x[i]);
    y[i] = __float2half(val > 0.f ? gamma * val : scale * (exp(val) - 1.f));
  }
}

__global__ void _Selu(
    const int N,
    const float scale,
    const float gamma,
    const half2* x,
    half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(
        val.x > 0.f ? gamma * val.x : scale * (exp(val.x) - 1.f),
        val.y > 0.f ? gamma * val.y : scale * (exp(val.y) - 1.f));
  }
}

template <typename T>
__global__ void _SeluGrad(
    const int N,
    const T scale,
    const T gamma,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (__ldg(y + i) > 0 ? gamma : (scale + __ldg(y + i)));
  }
}

__global__ void _SeluGrad(
    const int N,
    const float scale,
    const float gamma,
    const half* dy,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] =
        __float2half(__half2float(dy[i]) * (val > 0.f ? gamma : (scale + val)));
  }
} // SeluGrad

__global__ void _SeluGrad(
    const int N,
    const float scale,
    const float gamma,
    const half2* dy,
    const half2* y,
    half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(
        grad.x * (val.x > 0.f ? gamma : (scale + val.x)),
        grad.y * (val.y > 0.f ? gamma : (scale + val.y)));
  }
} // SeluGrad

} // namespace

template <>
void Selu<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float gamma,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _Selu<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Selu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

template <>
void SeluGrad<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float gamma,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _SeluGrad<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _SeluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // SeluGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                   \
  template <>                                                       \
  void Selu<T, CUDAContext>(                                        \
      const int N,                                                  \
      const float alpha,                                            \
      const float gamma,                                            \
      const T* x,                                                   \
      T* y,                                                         \
      CUDAContext* ctx) {                                           \
    _Selu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, T(alpha * gamma), T(gamma), x, y);                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void SeluGrad<T, CUDAContext>(                                        \
      const int N,                                                      \
      const float alpha,                                                \
      const float gamma,                                                \
      const T* dy,                                                      \
      const T* y,                                                       \
      T* dx,                                                            \
      CUDAContext* ctx) {                                               \
    _SeluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, T(alpha * gamma), T(gamma), dy, y, dx);                      \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
