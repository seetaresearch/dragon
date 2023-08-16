#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

/*
 * Relu Kernels
 */

template <typename T>
__global__ void _Relu(const int N, const T alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > 0 ? __ldg(x + i) : __ldg(x + i) * alpha;
  }
}

template <>
__global__ void
_Relu<half>(const int N, const half alpha, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > kZero ? __ldg(x + i) : __ldg(x + i) * alpha;
  }
#else
  const float kAlpha = __half2float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? __ldg(x + i) : __float2half(val * kAlpha);
  }
#endif
}

template <>
__global__ void _Relu<nv_bfloat16>(
    const int N,
    const nv_bfloat16 alpha,
    const nv_bfloat16* x,
    nv_bfloat16* y) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > kZero ? __ldg(x + i) : __ldg(x + i) * alpha;
  }
#else
  const float kAlpha = __bfloat162float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __bfloat162float(x[i]);
    y[i] = val > 0.f ? x[i] : __float2bfloat16(val * kAlpha);
  }
#endif
}

/*
 * ReluN Kernels
 */

template <typename T>
__global__ void _ReluN(const int N, const T high, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > 0 ? __ldg(x + i) < high ? __ldg(x + i) : high : T(0);
  }
}

template <>
__global__ void
_ReluN<half>(const int N, const half high, const half* x, half* y) {
  const half kZero = __float2half(0.f);
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > kZero ? __hmin(__ldg(x + i), high) : kZero;
  }
#else
  const float kHigh = __half2float(high);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? (val < kHigh ? __ldg(x + i) : high) : kZero;
  }
#endif
}

template <>
__global__ void _ReluN<nv_bfloat16>(
    const int N,
    const nv_bfloat16 high,
    const nv_bfloat16* x,
    nv_bfloat16* y) {
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
#if __CUDA_ARCH__ >= 800
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > kZero ? __hmin(__ldg(x + i), high) : kZero;
  }
#else
  const float kHigh = __bfloat162float(high);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __bfloat162float(x[i]);
    y[i] = val > 0.f ? (val < kHigh ? x[i] : high) : kZero;
  }
#endif
}

/*
 * ReluGrad Kernels
 */

template <typename T>
__global__ void
_ReluGrad(const int N, const T alpha, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __ldg(y + i) > T(0) ? dy[i] : dy[i] * alpha;
  }
}

template <>
__global__ void _ReluGrad<half>(
    const int N,
    const half alpha,
    const half* dy,
    const half* y,
    half* dx) {
#if __CUDA_ARCH__ >= 530
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = y[i] > kZero ? dy[i] : dy[i] * alpha;
  }
#else
  const float kAlpha = __half2float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __half2float(y[i]) > 0.f
        ? dy[i]
        : __float2half(__half2float(dy[i]) * kAlpha);
  }
#endif
}

template <>
__global__ void _ReluGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16 alpha,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = y[i] > kZero ? dy[i] : dy[i] * alpha;
  }
#else
  const float kAlpha = __bfloat162float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __bfloat162float(y[i]) > 0.f
        ? dy[i]
        : __float2bfloat16(__bfloat162float(dy[i]) * kAlpha);
  }
#endif
}

/*
 * ReluNGrad Kernels
 */

template <typename T>
__global__ void
_ReluNGrad(const int N, const T high, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = (__ldg(y + i) > T(0) && __ldg(y + i) < high) ? dy[i] : T(0);
  }
}

template <>
__global__ void _ReluNGrad<half>(
    const int N,
    const half high,
    const half* dy,
    const half* y,
    half* dx) {
  const half kZero = __float2half(0.f);
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = (__ldg(y + i) > kZero && __ldg(y + i) < high) ? dy[i] : kZero;
  }
#else
  const float kHigh = __half2float(high);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] = (val > 0.f && val < kHigh) ? dy[i] : kZero;
  }
#endif
}

template <>
__global__ void _ReluNGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16 high,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
#if __CUDA_ARCH__ >= 800
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = (__ldg(y + i) > kZero && __ldg(y + i) < high) ? dy[i] : kZero;
  }
#else
  const float kHigh = __bfloat162float(high);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __bfloat162float(y[i]);
    dx[i] = (val > 0.f && val < kHigh) ? dy[i] : kZero;
  }
#endif
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                   \
  template <>                                                             \
  void name<T, CUDAContext>(                                              \
      const int N, const float arg, const T* x, T* y, CUDAContext* ctx) { \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
        N,                                                                \
        convert::To<math::Traits<T>::scalar_type>(arg),                   \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),         \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));              \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const float arg,                                                \
      const T* dy,                                                    \
      const T* y,                                                     \
      T* dx,                                                          \
      CUDAContext* ctx) {                                             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                            \
        convert::To<math::Traits<T>::scalar_type>(arg),               \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),    \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(y),     \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));         \
  }

DEFINE_KERNEL_LAUNCHER(Relu, float16);
DEFINE_KERNEL_LAUNCHER(Relu, bfloat16);
DEFINE_KERNEL_LAUNCHER(Relu, float);
DEFINE_KERNEL_LAUNCHER(Relu, double);
DEFINE_KERNEL_LAUNCHER(ReluN, float16);
DEFINE_KERNEL_LAUNCHER(ReluN, bfloat16);
DEFINE_KERNEL_LAUNCHER(ReluN, float);
DEFINE_KERNEL_LAUNCHER(ReluN, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
