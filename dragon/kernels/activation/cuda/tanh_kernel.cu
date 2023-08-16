#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _Tanh(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = tanh(convert::To<AccT>(x[i]));
  }
}

template <typename T>
__global__ void _TanhGrad(const int N, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (T(1) - math::utils::Sqr(y[i]));
  }
}

template <>
__global__ void
_TanhGrad<half>(const int N, const half* dy, const half* y, half* dx) {
#if __CUDA_ARCH__ >= 530
  const half kOne = __float2half(1.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (kOne - math::utils::Sqr(y[i]));
  }
#else
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __half2float(dy[i]) * (1.f - math::utils::Sqr(__half2float(y[i])));
  }
#endif
}

template <>
__global__ void _TanhGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kOne = __float2bfloat16(1.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (kOne - math::utils::Sqr(y[i]));
  }
#else
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __bfloat162float(dy[i]) *
        (1.f - math::utils::Sqr(__bfloat162float(y[i])));
  }
#endif
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void Tanh<T, CUDAContext>(const int N, const T* x, T* y, CUDAContext* ctx) { \
    _Tanh<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type>     \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(             \
            N,                                                                 \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),          \
            reinterpret_cast<math::Traits<T>::scalar_type*>(y));               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void TanhGrad<T, CUDAContext>(                                        \
      const int N, const T* dy, const T* y, T* dx, CUDAContext* ctx) {  \
    _TanhGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                              \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),      \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(y),       \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));           \
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
