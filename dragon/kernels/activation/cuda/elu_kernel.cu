#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _Elu(const int N, const AccT alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = math::utils::LDGC<AccT>(x + i);
    y[i] = v > 0.f ? math::utils::LDG(x + i)
                   : convert::To<T>(alpha * (exp(v) - AccT(1)));
  }
}

template <typename T>
__global__ void
_EluGrad(const int N, const T alpha, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __ldg(y + i) > T(0) ? dy[i] : dy[i] * (alpha + __ldg(y + i));
  }
}

template <>
__global__ void _EluGrad<half>(
    const int N,
    const half alpha,
    const half* dy,
    const half* y,
    half* dx) {
#if __CUDA_ARCH__ >= 530
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __ldg(y + i) > kZero ? dy[i] : dy[i] * (alpha + __ldg(y + i));
  }
#else
  const float kAlpha = __half2float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float v = __half2float(y[i]);
    dx[i] = v > 0.f ? dy[i] : __float2half(__half2float(dy[i]) * (kAlpha + v));
  }
#endif
}

template <>
__global__ void _EluGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16 alpha,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __ldg(y + i) > kZero ? dy[i] : dy[i] * (alpha + __ldg(y + i));
  }
#else
  const float kAlpha = __bfloat162float(alpha);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float v = __bfloat162float(y[i]);
    dx[i] = v > 0.f ? dy[i]
                    : __float2bfloat16(__bfloat162float(dy[i]) * (kAlpha + v));
  }
#endif
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void Elu<T, CUDAContext>(                                                 \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    _Elu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(          \
        N,                                                                  \
        convert::To<math::Traits<T>::accumulator_type>(alpha),              \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),           \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                 \
  template <>                                                          \
  void EluGrad<T, CUDAContext>(                                        \
      const int N,                                                     \
      const float alpha,                                               \
      const T* dy,                                                     \
      const T* y,                                                      \
      T* dx,                                                           \
      CUDAContext* ctx) {                                              \
    _EluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                             \
        convert::To<math::Traits<T>::scalar_type>(alpha),              \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),     \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(y),      \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));          \
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
