#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

/*
 * Sigmoid Kernels
 */

template <typename T, typename AccT>
__global__ void _Sigmoid(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = AccT(1) / (AccT(1) + exp(-convert::To<AccT>(x[i])));
  }
}

template <typename T>
__global__ void
_HardSigmoid(const int N, const T alpha, const T beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = max(T(0), min(fma(x[i], alpha, beta), T(1)));
  }
}

template <>
__global__ void _HardSigmoid<half>(
    const int N,
    const half alpha,
    const half beta,
    const half* x,
    half* y) {
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __hfma_sat(x[i], alpha, beta);
  }
#else
  const float kAlpha = __half2float(alpha);
  const float kBeta = __half2float(beta);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = max(min(fmaf(__half2float(x[i]), kAlpha, kBeta), 1.f), 0.f);
  }
#endif
}

template <>
__global__ void _HardSigmoid<nv_bfloat16>(
    const int N,
    const nv_bfloat16 alpha,
    const nv_bfloat16 beta,
    const nv_bfloat16* x,
    nv_bfloat16* y) {
#if __CUDA_ARCH__ >= 800
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __hfma_sat(x[i], alpha, beta);
  }
#else
  const float kAlpha = __bfloat162float(alpha);
  const float kBeta = __bfloat162float(beta);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = max(min(fmaf(__bfloat162float(x[i]), kAlpha, kBeta), 1.f), 0.f);
  }
#endif
}

/*
 * SigmoidGrad Kernels
 */

template <typename T>
__global__ void _SigmoidGrad(const int N, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * __ldg(y + i) * (T(1) - __ldg(y + i));
  }
}

template <>
__global__ void
_SigmoidGrad<half>(const int N, const half* dy, const half* y, half* dx) {
#if __CUDA_ARCH__ >= 530
  const half kOne = __float2half(1.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * __ldg(y + i) * (kOne - __ldg(y + i));
  }
#else
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] = __half2float(dy[i]) * val * (1.f - val);
  }
#endif
}

template <>
__global__ void _SigmoidGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kOne = __float2bfloat16(1.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * __ldg(y + i) * (kOne - __ldg(y + i));
  }
#else
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __bfloat162float(y[i]);
    dx[i] = __bfloat162float(dy[i]) * val * (1.f - val);
  }
#endif
}

template <typename T>
__global__ void
_HardSigmoidGrad(const int N, const T alpha, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = (__ldg(y + i) > T(0) && __ldg(y + i) < T(1)) ? dy[i] * alpha : T(0);
  }
}

template <>
__global__ void _HardSigmoidGrad<half>(
    const int N,
    const half alpha,
    const half* dy,
    const half* y,
    half* dx) {
  const half kZero = __float2half(0.f);
#if __CUDA_ARCH__ >= 530
  const half kOne = __float2half(1.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __ldg(y + i) > kZero && __ldg(y + i) < kOne ? dy[i] * alpha : kZero;
  }
#else
  const float kAlpha = __half2float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] = val > 0.f && val < 1.f ? __float2half(__half2float(dy[i]) * kAlpha)
                                   : kZero;
  }
#endif
}

template <>
__global__ void _HardSigmoidGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16 alpha,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kOne = __float2bfloat16(1.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __ldg(y + i) > kZero && __ldg(y + i) < kOne ? dy[i] * alpha : kZero;
  }
#else
  const float kAlpha = __bfloat162float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __bfloat162float(y[i]);
    dx[i] = val > 0.f && val < 1.f
        ? __float2bfloat16(__bfloat162float(dy[i]) * kAlpha)
        : kZero;
  }
#endif
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void Sigmoid<T, CUDAContext>(                                               \
      const int N, const T* x, T* y, CUDAContext* ctx) {                      \
    _Sigmoid<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(            \
            N,                                                                \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),         \
            reinterpret_cast<math::Traits<T>::scalar_type*>(y));              \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                     \
  template <>                                                              \
  void SigmoidGrad<T, CUDAContext>(                                        \
      const int N, const T* dy, const T* y, T* dx, CUDAContext* ctx) {     \
    _SigmoidGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                                 \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),         \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(y),          \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));              \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void HardSigmoid<T, CUDAContext>(                                        \
      const int N,                                                         \
      const float alpha,                                                   \
      const float beta,                                                    \
      const T* x,                                                          \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    _HardSigmoid<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                                 \
        convert::To<math::Traits<T>::scalar_type>(alpha),                  \
        convert::To<math::Traits<T>::scalar_type>(beta),                   \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),          \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                         \
  template <>                                                                  \
  void HardSigmoidGrad<T, CUDAContext>(                                        \
      const int N,                                                             \
      const float alpha,                                                       \
      const T* dy,                                                             \
      const T* y,                                                              \
      T* dx,                                                                   \
      CUDAContext* ctx) {                                                      \
    _HardSigmoidGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                                     \
        convert::To<math::Traits<T>::scalar_type>(alpha),                      \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),             \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(y),              \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));                  \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
