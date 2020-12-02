#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Swish(const int nthreads, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) / (T(1) + exp(-__ldg(x + i)));
#else
    y[i] = x[i] / (T(1) + exp(-x[i]));
#endif
  }
}

template <>
__global__ void _Swish<half>(const int nthreads, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __float2half(
        __half2float(__ldg(x + i)) / (1.f + exp(-__half2float(__ldg(x + i)))));
#else
    y[i] = __float2half(__half2float(x[i]) / (1.f + exp(-__half2float(x[i]))));
#endif
  }
}

template <typename T>
__global__ void
_SwishGrad(const int nthreads, const T* dy, const T* x, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] =
        dy[i] * (__ldg(y + i) + (T(1) - __ldg(y + i)) / (T(1) + exp(-x[i])));
#else
    dx[i] = dy[i] * (y[i] + (T(1) - y[i]) / (T(1) + exp(-x[i])));
#endif
  }
}

template <>
__global__ void _SwishGrad<half>(
    const int nthreads,
    const half* dy,
    const half* x,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = __float2half(
        __half2float(dy[i]) *
        (__half2float(__ldg(y + i)) +
         (1.f - __half2float(__ldg(y + i))) /
             (1.f + exp(-__half2float(x[i])))));
#else
    dx[i] = __float2half(
        __half2float(dy[i]) *
        (__half2float(y[i]) +
         (1.f - __half2float(y[i])) / (1.f + exp(-__half2float(x[i])))));
#endif
  }
} // SwishGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Swish<float16, CUDAContext>(
    const int count,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _Swish<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
}

template <>
void SwishGrad<float16, CUDAContext>(
    const int count,
    const float16* dy,
    const float16* x,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  _SwishGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<const half*>(x),
      reinterpret_cast<const half*>(y),
      reinterpret_cast<half*>(dx));
} // SwishGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void Swish<T, CUDAContext>(                                            \
      const int count, const T* x, T* y, CUDAContext* ctx) {             \
    _Swish<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, x, y);                                                    \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void SwishGrad<T, CUDAContext>(                                            \
      const int count,                                                       \
      const T* dy,                                                           \
      const T* x,                                                            \
      const T* y,                                                            \
      T* dx,                                                                 \
      CUDAContext* ctx) {                                                    \
    _SwishGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, dy, x, y, dx);                                                \
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
