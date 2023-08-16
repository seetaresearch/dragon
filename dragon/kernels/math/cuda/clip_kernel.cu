#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_Clip(const int N, const T low, const T high, const T* x, T* y) {
  const auto clamp = math::ClampFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = clamp(x[i], low, high);
  }
}

template <typename T>
__global__ void _ClipGrad(
    const int N,
    const T low,
    const T high,
    const T* dy,
    const T* x,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T val = x[i];
    dx[i] = (val < low || val > high) ? T(0) : dy[i];
  }
}

template <>
__global__ void _ClipGrad<half>(
    const int N,
    const half low,
    const half high,
    const half* dy,
    const half* x,
    half* dx) {
  const half kZero = __float2half(0.f);
#if __CUDA_ARCH__ < 530
  const float kLow = __half2float(low);
  const float kHigh = __half2float(high);
#endif
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    const half val = x[i];
    dx[i] = (val < low || val > high) ? kZero : dy[i];
#else
    const float val = __half2float(x[i]);
    dx[i] = (val < kLow || val > kHigh) ? kZero : dy[i];
#endif
  }
}

template <>
__global__ void _ClipGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16 low,
    const nv_bfloat16 high,
    const nv_bfloat16* dy,
    const nv_bfloat16* x,
    nv_bfloat16* dx) {
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
#if __CUDA_ARCH__ < 800
  const float kLow = __bfloat162float(low);
  const float kHigh = __bfloat162float(high);
#endif
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    const nv_bfloat16 val = x[i];
    dx[i] = (val < low || val > high) ? kZero : dy[i];
#else
    const float val = __bfloat162float(x[i]);
    dx[i] = (val < kLow || val > kHigh) ? kZero : dy[i];
#endif
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                   \
  template <>                                                       \
  void Clip<T, CUDAContext>(                                        \
      const int N,                                                  \
      const float low,                                              \
      const float high,                                             \
      const T* x,                                                   \
      T* y,                                                         \
      CUDAContext* ctx) {                                           \
    _Clip<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                          \
        convert::To<math::Traits<T>::scalar_type>(low),             \
        convert::To<math::Traits<T>::scalar_type>(high),            \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),   \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));        \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void ClipGrad<T, CUDAContext>(                                        \
      const int N,                                                      \
      const float low,                                                  \
      const float high,                                                 \
      const T* dy,                                                      \
      const T* x,                                                       \
      T* dx,                                                            \
      CUDAContext* ctx) {                                               \
    _ClipGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                              \
        convert::To<math::Traits<T>::scalar_type>(low),                 \
        convert::To<math::Traits<T>::scalar_type>(high),                \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),      \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),       \
        reinterpret_cast<math::Traits<T>::scalar_type*>(dx));           \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
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
