#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Elu(const int nthreads, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] =
        __ldg(x + i) > T(0) ? __ldg(x + i) : alpha * (exp(__ldg(x + i)) - T(1));
#else
    y[i] = x[i] > T(0) ? x[i] : alpha * (exp(x[i]) - T(1));
#endif
  }
}

template <>
__global__ void
_Elu<half>(const int nthreads, const float alpha, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? __ldg(x + i) : __float2half(alpha * (exp(val) - 1.f));
#endif
  }
}

template <>
__global__ void
_Elu<half2>(const int nthreads, const float alpha, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(
        val.x > 0.f ? val.x : alpha * (exp(val.x) - 1.f),
        val.y > 0.f ? val.y : alpha * (exp(val.y) - 1.f));
#endif
  }
}

template <typename T>
__global__ void _EluGrad(
    const int nthreads,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = dy[i] * (__ldg(y + i) > T(0) ? T(1) : alpha + __ldg(y + i));
#else
    dx[i] = dy[i] * (y[i] > T(0) ? T(1) : (alpha + y[i]));
#endif
  }
}

template <>
__global__ void _EluGrad<half>(
    const int nthreads,
    const float alpha,
    const half* dy,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    const float val = __half2float(y[i]);
    dx[i] = __hmul(dy[i], __float2half(val > 0.f ? 1.f : (alpha + val)));
#endif
  }
} // EluGrad

template <>
__global__ void _EluGrad<half2>(
    const int nthreads,
    const float alpha,
    const half2* dy,
    const half2* y,
    half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    const float2 val = __half22float2(y[i]);
    dx[i] = __hmul2(
        dy[i],
        __floats2half2_rn(
            val.x > 0.f ? 1.f : (alpha + val.x),
            val.y > 0.f ? 1.f : (alpha + val.y)));
#endif
  }
} // EluGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Elu<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _Elu<<<CUDA_BLOCKS(count >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count >> 1,
        alpha,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Elu<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        alpha,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

template <>
void EluGrad<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _EluGrad<<<CUDA_BLOCKS(count >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count >> 1,
        alpha,
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _EluGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        alpha,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // EluGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void Elu<T, CUDAContext>(                                            \
      const int count,                                                 \
      const float alpha,                                               \
      const T* x,                                                      \
      T* y,                                                            \
      CUDAContext* ctx) {                                              \
    _Elu<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, alpha, x, y);                                           \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                     \
  template <>                                                              \
  void EluGrad<T, CUDAContext>(                                            \
      const int count,                                                     \
      const float alpha,                                                   \
      const T* dy,                                                         \
      const T* y,                                                          \
      T* dx,                                                               \
      CUDAContext* ctx) {                                                  \
    _EluGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, alpha, dy, y, dx);                                          \
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
