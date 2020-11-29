#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/conversions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Selu(
    const int nthreads,
    const float alphaXgamma,
    const float gamma,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) > 0 ? gamma * __ldg(x + i)
                            : alphaXgamma * (exp(__ldg(x + i)) - 1);
#else
    y[i] = x[i] > 0 ? gamma * x[i] : alphaXgamma * (exp(x[i]) - 1);
#endif
  }
}

template <>
__global__ void _Selu<half>(
    const int nthreads,
    const float alphaXgamma,
    const float gamma,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float val = __half2float(x[i]);
    y[i] =
        __float2half(val > 0.f ? gamma * val : alphaXgamma * (exp(val) - 1.f));
  }
}

template <>
__global__ void _Selu<half2>(
    const int nthreads,
    const float alphaXgamma,
    const float gamma,
    const half2* x,
    half2* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(
        val.x > 0.f ? gamma * val.x : alphaXgamma * (exp(val.x) - 1.f),
        val.y > 0.f ? gamma * val.y : alphaXgamma * (exp(val.y) - 1.f));
  }
}

template <typename T>
__global__ void _SeluGrad(
    const int nthreads,
    const float alphaXgamma,
    const float gamma,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = dy[i] * (__ldg(y + i) > 0 ? gamma : (alphaXgamma + __ldg(y + i)));
#else
    dx[i] = dy[i] * (y[i] > 0 ? gamma : (alphaXgamma + y[i]));
#endif
  }
}

template <>
__global__ void _SeluGrad<half>(
    const int nthreads,
    const float alphaXgamma,
    const float gamma,
    const half* dy,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float val = __half2float(y[i]);
    dx[i] = __float2half(
        __half2float(dy[i]) * (val > 0.f ? gamma : (alphaXgamma + val)));
  }
} // SeluGrad

template <>
__global__ void _SeluGrad<half2>(
    const int nthreads,
    const float alphaXgamma,
    const float gamma,
    const half2* dy,
    const half2* y,
    half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(
        grad.x * (val.x > 0.f ? gamma : (alphaXgamma + val.x)),
        grad.y * (val.y > 0.f ? gamma : (alphaXgamma + val.y)));
  }
} // SeluGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Selu<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float gamma,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _Selu<<<CUDA_BLOCKS(count >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count >> 1,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Selu<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

template <>
void SeluGrad<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float gamma,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _SeluGrad<<<CUDA_BLOCKS(count >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count >> 1,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _SeluGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        alpha * gamma,
        gamma,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // SeluGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                       \
  template <>                                                           \
  void Selu<T, CUDAContext>(                                            \
      const int count,                                                  \
      const float alpha,                                                \
      const float gamma,                                                \
      const T* x,                                                       \
      T* y,                                                             \
      CUDAContext* ctx) {                                               \
    _Selu<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, alpha * gamma, gamma, x, y);                             \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                      \
  template <>                                                               \
  void SeluGrad<T, CUDAContext>(                                            \
      const int count,                                                      \
      const float alpha,                                                    \
      const float gamma,                                                    \
      const T* dy,                                                          \
      const T* y,                                                           \
      T* dx,                                                                \
      CUDAContext* ctx) {                                                   \
    _SeluGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, alpha * gamma, gamma, dy, y, dx);                            \
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
