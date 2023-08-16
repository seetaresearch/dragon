#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

/*
 * Swish Kernels
 */

template <typename T, typename AccT>
__global__ void _Silu(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    y[i] = v / (AccT(1) + exp(-v));
  }
}

template <typename T, typename AccT>
__global__ void _HardSwish(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) *
        max(min(fma(__ldg(x + i), T(0.166667), T(0.5)), T(1)), T(0));
  }
}

template <>
__global__ void _HardSwish<half, float>(const int N, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530
  const half kAlpha = __float2half(0.166667f);
  const half kBeta = __float2half(0.5f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) * __hfma_sat(__ldg(x + i), kAlpha, kBeta);
  }
#else
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float v = __half2float(x[i]);
    y[i] = v * max(min(fmaf(v, 0.166667f, 0.5f), 1.f), 0.f);
  }
#endif
}

template <>
__global__ void _HardSwish<nv_bfloat16, float>(
    const int N,
    const nv_bfloat16* x,
    nv_bfloat16* y) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kAlpha = __float2bfloat16(0.166667f);
  const nv_bfloat16 kBeta = __float2bfloat16(0.5f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) * __hfma_sat(__ldg(x + i), kAlpha, kBeta);
  }
#else
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float v = __bfloat162float(x[i]);
    y[i] = v * max(min(fmaf(v, 0.166667f, 0.5f), 1.f), 0.f);
  }
#endif
}

/*
 * SwishGrad Kernels
 */

template <typename T, typename AccT>
__global__ void _SiluGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    const AccT s = AccT(1) / (AccT(1) + exp(-v));
    dx[i] = convert::To<AccT>(dy[i]) * s * (v + AccT(1) - v * s);
  }
}

template <typename T, typename AccT>
__global__ void _HardSwishGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    dx[i] = v < AccT(-3) ? convert::To<T>(AccT(0))
        : v < AccT(3)
        ? T(convert::To<AccT>(dy[i]) * fma(v, AccT(0.333333), AccT(0.5)))
        : dy[i];
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                        \
  template <>                                                                  \
  void name<T, CUDAContext>(const int N, const T* x, T* y, CUDAContext* ctx) { \
    _##name<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type>   \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(             \
            N,                                                                 \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),          \
            reinterpret_cast<math::Traits<T>::scalar_type*>(y));               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                 \
  template <>                                                                \
  void name<T, CUDAContext>(                                                 \
      const int N, const T* dy, const T* x, T* dx, CUDAContext* ctx) {       \
    _##name<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(           \
            N,                                                               \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),       \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),        \
            reinterpret_cast<math::Traits<T>::scalar_type*>(dx));            \
  }

DEFINE_KERNEL_LAUNCHER(Silu, float16);
DEFINE_KERNEL_LAUNCHER(Silu, bfloat16);
DEFINE_KERNEL_LAUNCHER(Silu, float);
DEFINE_KERNEL_LAUNCHER(Silu, double);
DEFINE_KERNEL_LAUNCHER(HardSwish, float16);
DEFINE_KERNEL_LAUNCHER(HardSwish, bfloat16);
DEFINE_KERNEL_LAUNCHER(HardSwish, float);
DEFINE_KERNEL_LAUNCHER(HardSwish, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
