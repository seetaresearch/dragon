#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_HardSwish(const int nthreads, const T alpha, const T beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) * max(T(0), min(T(1), fma(__ldg(x + i), alpha, beta)));
#else
    y[i] = x[i] * max(T(0), min(T(1), fma(x[i], alpha, beta)));
#endif
  }
}

__global__ void _HardSwish(
    const int nthreads,
    const float alpha,
    const float beta,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __float2half(
        __half2float(__ldg(x + i)) *
        max(0.f, min(1.f, fma(__half2float(__ldg(x + i)), alpha, beta))));
#else
    y[i] = __float2half(
        __half2float(x[i]) *
        max(0.f, min(1.f, fma(__half2float(x[i]), alpha, beta))));
#endif
  }
}

template <typename T>
__global__ void _HardSwishGrad(
    const int nthreads,
    const T alpha,
    const T beta,
    const T* dy,
    const T* x,
    T* dx) {
  const T bound = beta / alpha;
  const T alpha2x = alpha * T(2);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = (__ldg(x + i) < -bound)
        ? T(0)
        : (__ldg(x + i) < bound) ? dy[i] * fma(__ldg(x + i), alpha2x, beta)
                                 : dy[i];
#else
    dx[i] = (x[i] < -bound)
        ? T(0)
        : (x[i] < bound) ? dy[i] * fma(x[i], alpha2x, beta) : dy[i];
#endif
  }
}

__global__ void _HardSwishGrad(
    const int nthreads,
    const float alpha,
    const float beta,
    const half* dy,
    const half* x,
    half* dx) {
  const float bound = beta / alpha;
  const float alpha2x = alpha * 2.f;
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float val = __half2float(x[i]);
    dx[i] = (val < -bound) ? kZero
                           : (val < bound)
            ? __float2half(__half2float(dy[i]) * fma(val, alpha2x, beta))
            : dy[i];
  }
} // HardSwishGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void HardSwish<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float beta,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _HardSwish<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      alpha,
      beta,
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y));
}

template <>
void HardSwishGrad<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float beta,
    const float16* dy,
    const float16* x,
    float16* dx,
    CUDAContext* ctx) {
  _HardSwishGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      alpha,
      beta,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(dx));
} // HardSwishGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void HardSwish<T, CUDAContext>(                                            \
      const int count,                                                       \
      const float alpha,                                                     \
      const float beta,                                                      \
      const T* x,                                                            \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    _HardSwish<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, T(alpha), T(beta), x, y);                                     \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                              \
  template <>                                                       \
  void HardSwishGrad<T, CUDAContext>(                               \
      const int count,                                              \
      const float alpha,                                            \
      const float beta,                                             \
      const T* dy,                                                  \
      const T* x,                                                   \
      T* dx,                                                        \
      CUDAContext* ctx) {                                           \
    _HardSwishGrad<<<                                               \
        CUDA_BLOCKS(count),                                         \
        CUDA_THREADS,                                               \
        0,                                                          \
        ctx->cuda_stream()>>>(count, T(alpha), T(beta), dy, x, dx); \
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
