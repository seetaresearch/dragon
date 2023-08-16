#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_Selu(const int N, const AccT scale, const AccT gamma, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    y[i] = v > 0.f ? gamma * v : scale * (exp(v) - AccT(1));
  }
}

template <typename T>
__global__ void _SeluGrad(
    const int N,
    const T scale,
    const T gamma,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (__ldg(y + i) > 0 ? gamma : scale + __ldg(y + i));
  }
}

template <>
__global__ void _SeluGrad<half>(
    const int N,
    const half scale,
    const half gamma,
    const half* dy,
    const half* y,
    half* dx) {
#if __CUDA_ARCH__ >= 530
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (__ldg(y + i) > kZero ? gamma : scale + __ldg(y + i));
  }
#else
  const float kGamma = __half2float(gamma);
  const float kScale = __half2float(scale);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float v = __half2float(y[i]);
    dx[i] = __half2float(dy[i]) * (v > 0.f ? kGamma : kScale + v);
  }
#endif
}

template <>
__global__ void _SeluGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16 scale,
    const nv_bfloat16 gamma,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (__ldg(y + i) > kZero ? gamma : scale + __ldg(y + i));
  }
#else
  const float kGamma = __bfloat162float(gamma);
  const float kScale = __bfloat162float(scale);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float v = __bfloat162float(y[i]);
    dx[i] = __bfloat162float(dy[i]) * (v > 0.f ? kGamma : kScale + v);
  }
#endif
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void Selu<T, CUDAContext>(                                           \
      const int N,                                                     \
      const float alpha,                                               \
      const float gamma,                                               \
      const T* x,                                                      \
      T* y,                                                            \
      CUDAContext* ctx) {                                              \
    _Selu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
        N,                                                             \
        convert::To<math::Traits<T>::accumulator_type>(alpha * gamma), \
        convert::To<math::Traits<T>::accumulator_type>(gamma),         \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),      \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));           \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void SeluGrad<T, CUDAContext>(                                        \
      const int N,                                                      \
      const float alpha,                                                \
      const float gamma,                                                \
      const T* dy,                                                      \
      const T* y,                                                       \
      T* dx,                                                            \
      CUDAContext* ctx) {                                               \
    _SeluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                              \
        convert::To<math::Traits<T>::scalar_type>(alpha * gamma),       \
        convert::To<math::Traits<T>::scalar_type>(gamma),               \
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
