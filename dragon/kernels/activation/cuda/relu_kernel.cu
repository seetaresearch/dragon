#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/conversions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Relu(const int N, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > 0 ? __ldg(x + i) : __ldg(x + i) * alpha;
  }
}

template <>
__global__ void
_Relu<half>(const int N, const float alpha, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? __ldg(x + i) : __float2half(val * alpha);
  }
}

template <>
__global__ void
_Relu<half2>(const int N, const float alpha, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(
        val.x > 0.f ? val.x : val.x * alpha,
        val.y > 0.f ? val.y : val.y * alpha);
  }
}

template <typename T>
__global__ void _ReluN(const int N, const T max_value, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > 0
        ? (__ldg(x + i) < max_value ? __ldg(x + i) : max_value)
        : T(0);
  }
}

template <>
__global__ void
_ReluN<half>(const int N, const half max_value, const half* x, half* y) {
  const half kZero = __float2half(0.f);
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __hgt(__ldg(x + i), kZero)
        ? (__hlt(__ldg(x + i), max_value) ? __ldg(x + i) : max_value)
        : kZero;
  }
#else
  const float kMax = __half2float(max_value);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? ((val < kMax) ? __ldg(x + i) : max_value) : kZero;
  }
#endif
}

__global__ void
_ReluNHalf2(const int N, const half max_value, const half2* x, half2* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
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
#else
    const float kMax = __half2float(max_value);
    CUDA_1D_KERNEL_LOOP(i, N) {
      const float2 val = __half22float2(__ldg(x + i));
      y[i] = __halves2half2(
          val.x > 0.f ? ((val.x < kMax) ? __low2half(__ldg(x + i)) : max_value)
                      : kZero,
          val.y > 0.f ? ((val.y < kMax) ? __high2half(__ldg(x + i)) : max_value)
                      : kZero);
    }
#endif
  }
}

template <typename T>
__global__ void
_ReluGrad(const int N, const float alpha, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * ((__ldg(y + i) > 0) + alpha * (__ldg(y + i) <= 0));
  }
}

template <>
__global__ void _ReluGrad<half>(
    const int N,
    const float alpha,
    const half* dy,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] = __float2half(
        __half2float(dy[i]) * ((val > 0.f) + alpha * (val <= 0.f)));
  }
} // ReluGrad

template <>
__global__ void _ReluGrad<half2>(
    const int N,
    const float alpha,
    const half2* dy,
    const half2* y,
    half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(
        grad.x * ((val.x > 0.f) + alpha * (val.x <= 0.f)),
        grad.y * ((val.y > 0.f) + alpha * (val.y <= 0.f)));
  }
} // ReluGrad

template <typename T>
__global__ void
_ReluNGrad(const int N, const T max_value, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = (__ldg(y + i) > 0 && __ldg(y + i) < max_value) ? dy[i] : T(0);
  }
}

template <>
__global__ void _ReluNGrad<half>(
    const int N,
    const half max_value,
    const half* dy,
    const half* y,
    half* dx) {
  const half kZero = __float2half(0.f);
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = (__hgt(__ldg(y + i), kZero) && __hlt(__ldg(y + i), max_value))
        ? dy[i]
        : kZero;
  }
#else
  const float kMax = __half2float(max_value);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] = (val > 0.f && val < kMax) ? dy[i] : kZero;
  }
#endif
}

template <>
__global__ void _ReluNGrad<half2>(
    const int N,
    const half2 max_value,
    const half2* dy,
    const half2* y,
    half2* dx) {
#if __CUDA_ARCH__ >= 530
  const half2 kZero = __float2half2_rn(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __hmul2(
        __hmul2(__hgt2(__ldg(y + i), kZero), __hlt2(__ldg(y + i), max_value)),
        dy[i]);
  }
#else
  const half kZero = __float2half(0.f);
  const float kMax = __half2float(__low2half(max_value));
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(y[i]);
    dx[i] = __halves2half2(
        (val.x > 0.f && val.x < kMax) ? __low2half(__ldg(dy + i)) : kZero,
        (val.y > 0.f && val.y < kMax) ? __high2half(__ldg(dy + i)) : kZero);
  }
#endif
}

} // namespace

template <>
void Relu<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _Relu<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        alpha,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Relu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, alpha, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
}

template <>
void ReluN<float16, CUDAContext>(
    const int N,
    const float max_value,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _ReluNHalf2<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        convert::To<half>(max_value),
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _ReluN<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        convert::To<half>(max_value),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

template <>
void ReluGrad<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _ReluGrad<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        alpha,
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _ReluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        alpha,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // ReluGrad

template <>
void ReluNGrad<float16, CUDAContext>(
    const int N,
    const float max_value,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _ReluNGrad<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        convert::To<half2>(max_value),
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _ReluNGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        convert::To<half>(max_value),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // ReluNGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void Relu<T, CUDAContext>(                                                \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    _Relu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(         \
        N, convert::To<T>(alpha), x, y);                                    \
  }                                                                         \
  template <>                                                               \
  void ReluN<T, CUDAContext>(                                               \
      const int N,                                                          \
      const float max_value,                                                \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    _ReluN<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
        N, convert::To<T>(max_value), x, y);                                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                   \
  template <>                                                            \
  void ReluGrad<T, CUDAContext>(                                         \
      const int N,                                                       \
      const float alpha,                                                 \
      const T* dy,                                                       \
      const T* y,                                                        \
      T* dx,                                                             \
      CUDAContext* ctx) {                                                \
    _ReluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
        N, convert::To<T>(alpha), dy, y, dx);                            \
  }                                                                      \
  template <>                                                            \
  void ReluNGrad<T, CUDAContext>(                                        \
      const int N,                                                       \
      const float max_value,                                             \
      const T* dy,                                                       \
      const T* y,                                                        \
      T* dx,                                                             \
      CUDAContext* ctx) {                                                \
    _ReluNGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, convert::To<T>(max_value), dy, y, dx);                        \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
