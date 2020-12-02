#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _HardSigmoid(
    const int nthreads,
    const T alpha,
    const T beta,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = max(T(0), min(T(1), fma(x[i], alpha, beta)));
  }
}

__global__ void _HardSigmoid(
    const int nthreads,
    const float alpha,
    const float beta,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] =
        __float2half(max(0.f, min(1.f, fma(__half2float(x[i]), alpha, beta))));
  }
}

template <typename T>
__global__ void _HardSigmoidGrad(
    const int nthreads,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] = (y[i] > T(0) && y[i] < T(1)) ? dy[i] * alpha : T(0);
  }
}

template <>
__global__ void _HardSigmoidGrad<half>(
    const int nthreads,
    const float alpha,
    const half* dy,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float val = __half2float(y[i]);
    dx[i] = __half2float(
        (val > 0.f && val < 1.f) ? __half2float(dy[i]) * alpha : 0.f);
  }
} // HardSigmoidGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void HardSigmoid<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float beta,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _HardSigmoid<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      alpha,
      beta,
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y));
}

template <>
void HardSigmoidGrad<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  _HardSigmoidGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      alpha,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<const half*>(y),
      reinterpret_cast<half*>(dx));
} // HardSigmoidGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void HardSigmoid<T, CUDAContext>(                                            \
      const int count,                                                         \
      const float alpha,                                                       \
      const float beta,                                                        \
      const T* x,                                                              \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    _HardSigmoid<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, T(alpha), T(beta), x, y);                                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                  \
  template <>                                           \
  void HardSigmoidGrad<T, CUDAContext>(                 \
      const int count,                                  \
      const float alpha,                                \
      const T* dy,                                      \
      const T* y,                                       \
      T* dx,                                            \
      CUDAContext* ctx) {                               \
    _HardSigmoidGrad<<<                                 \
        CUDA_BLOCKS(count),                             \
        CUDA_THREADS,                                   \
        0,                                              \
        ctx->cuda_stream()>>>(count, alpha, dy, y, dx); \
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
