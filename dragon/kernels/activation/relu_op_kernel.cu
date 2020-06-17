#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/cast.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Relu(const int nthreads, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) > 0 ? __ldg(x + i) : __ldg(x + i) * alpha;
#else
    y[i] = x[i] > 0 ? x[i] : x[i] * alpha;
#endif
  }
}

template <>
__global__ void
_Relu<half>(const int nthreads, const float alpha, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? __ldg(x + i) : __float2half(val * alpha);
#endif
  }
}

template <>
__global__ void
_Relu<half2>(const int nthreads, const float alpha, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(
        val.x > 0.f ? val.x : val.x * alpha,
        val.y > 0.f ? val.y : val.y * alpha);
#endif
  }
}

template <typename T>
__global__ void
_ReluN(const int nthreads, const T max_value, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) > 0
        ? (__ldg(x + i) < max_value ? __ldg(x + i) : max_value)
        : T(0);
#else
    y[i] = x[i] > 0 ? (x[i] < max_value ? x[i] : max_value) : T(0);
#endif
  }
}

template <>
__global__ void
_ReluN<half>(const int nthreads, const half max_value, const half* x, half* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __hgt(__ldg(x + i), kZero)
        ? (__hlt(__ldg(x + i), max_value) ? __ldg(x + i) : max_value)
        : kZero;
#endif
  }
}

__global__ void _ReluNHalf2(
    const int nthreads,
    const half max_value,
    const half2* x,
    half2* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __halves2half2(
        __hgt(__low2half(__ldg(x + i)), kZero)
            ? (__hlt(__low2half(__ldg(x + i)), max_value)
                   ? __low2half(__ldg(x + i))
                   : max_value)
            : kZero,
        __hgt(__high2half(__ldg(x + i)), kZero)
            ? (__hlt(__high2half(__ldg(x + i)), max_value)
                   ? __high2half(__ldg(x + i))
                   : max_value)
            : kZero);
#endif
  }
}

template <typename T>
__global__ void _ReluGrad(
    const int nthreads,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = __ldg(dy + i) * ((__ldg(y + i) > 0) + alpha * (__ldg(y + i) <= 0));
#else
    dx[i] = dy[i] * ((y[i] > 0) + alpha * (y[i] <= 0));
#endif
  }
}

template <>
__global__ void _ReluGrad<half>(
    const int nthreads,
    const float alpha,
    const half* dy,
    const half* y,
    half* dx) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = __hmul(
        dy[i],
        __float2half(
            __hgt(__ldg(y + i), kZero) + __hle(__ldg(y + i), kZero) * alpha));
#endif
  }
} // ReluGrad

template <>
__global__ void _ReluGrad<half2>(
    const int nthreads,
    const float alpha,
    const half2* dy,
    const half2* y,
    half2* dx) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = __hmul2(
        dy[i],
        __floats2half2_rn(
            __hgt(__low2half(__ldg(y + i)), kZero) +
                __hle(__low2half(__ldg(y + i)), kZero) * alpha,
            __hgt(__high2half(__ldg(y + i)), kZero) +
                __hle(__high2half(__ldg(y + i)), kZero) * alpha));
#endif
  }
} // ReluGrad

template <typename T>
__global__ void _ReluNGrad(
    const int nthreads,
    const T max_value,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = (__ldg(y + i) > 0 && __ldg(y + i) < max_value) ? dy[i] : T(0);
#else
    dx[i] = (y[i] > 0 && y[i] < max_value) ? dy[i] : T(0);
#endif
  }
}

template <>
__global__ void _ReluNGrad<half>(
    const int nthreads,
    const half max_value,
    const half* dy,
    const half* y,
    half* dx) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = (__hgt(__ldg(y + i), kZero) && __hlt(__ldg(y + i), max_value))
        ? dy[i]
        : kZero;
#endif
  }
} // ReluNGrad

template <>
__global__ void _ReluNGrad<half2>(
    const int nthreads,
    const half2 max_value,
    const half2* dy,
    const half2* y,
    half2* dx) {
  const half2 kZero = __float2half2_rn(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = __hmul2(
        __hmul2(__hgt2(__ldg(y + i), kZero), __hlt2(__ldg(y + i), max_value)),
        dy[i]);
#endif
  }
} // ReluNGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Relu<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _Relu<<<CUDA_BLOCKS(count >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count >> 1,
        alpha,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Relu<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        alpha,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

template <>
void ReluN<float16, CUDAContext>(
    const int count,
    const float max_value,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _ReluNHalf2<<<
        CUDA_BLOCKS(count >> 1),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        count >> 1,
        cast::to<half>(max_value),
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _ReluN<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        cast::to<half>(max_value),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

template <>
void ReluGrad<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _ReluGrad<<<CUDA_BLOCKS(count >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count >> 1,
        alpha,
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _ReluGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        alpha,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // ReluGrad

template <>
void ReluNGrad<float16, CUDAContext>(
    const int count,
    const float max_value,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((count & 1) == 0) {
    _ReluNGrad<<<
        CUDA_BLOCKS(count >> 1),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        count >> 1,
        cast::to<half2>(max_value),
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _ReluNGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        count,
        cast::to<half>(max_value),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // ReluNGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void Relu<T, CUDAContext>(                                             \
      const int count,                                                   \
      const float alpha,                                                 \
      const T* x,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    _Relu<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
        count, cast::to<T>(alpha), x, y);                                \
  }                                                                      \
  template <>                                                            \
  void ReluN<T, CUDAContext>(                                            \
      const int count,                                                   \
      const float max_value,                                             \
      const T* x,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    _ReluN<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, cast::to<T>(max_value), x, y);                            \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void ReluGrad<T, CUDAContext>(                                             \
      const int count,                                                       \
      const float alpha,                                                     \
      const T* dy,                                                           \
      const T* y,                                                            \
      T* dx,                                                                 \
      CUDAContext* ctx) {                                                    \
    _ReluGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
        count, cast::to<T>(alpha), dy, y, dx);                               \
  }                                                                          \
  template <>                                                                \
  void ReluNGrad<T, CUDAContext>(                                            \
      const int count,                                                       \
      const float max_value,                                                 \
      const T* dy,                                                           \
      const T* y,                                                            \
      T* dx,                                                                 \
      CUDAContext* ctx) {                                                    \
    _ReluNGrad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, cast::to<T>(max_value), dy, y, dx);                           \
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
