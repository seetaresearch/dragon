#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) __half2float(__ldg(x + i))

template <typename T>
__global__ void
_HardSwish(const int N, const T alpha, const T beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = LDG(x, i) * max(T(0), min(T(1), fma(LDG(x, i), alpha, beta)));
  }
}

__global__ void _HardSwish(
    const int N,
    const float alpha,
    const float beta,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(
        LDG2(x, i) * max(0.f, min(1.f, fma(LDG2(x, i), alpha, beta))));
  }
}

template <typename T>
__global__ void _HardSwishGrad(
    const int N,
    const T alpha,
    const T beta,
    const T* dy,
    const T* x,
    T* dx) {
  const T bound = beta / alpha;
  const T alpha2x = alpha * T(2);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = (LDG(x, i) < -bound)
        ? T(0)
        : (LDG(x, i) < bound) ? dy[i] * fma(LDG(x, i), alpha2x, beta) : dy[i];
  }
}

__global__ void _HardSwishGrad(
    const int N,
    const float alpha,
    const float beta,
    const half* dy,
    const half* x,
    half* dx) {
  const float bound = beta / alpha;
  const float alpha2x = alpha * 2.f;
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(x[i]);
    dx[i] = (val < -bound) ? kZero
                           : (val < bound)
            ? __float2half(__half2float(dy[i]) * fma(val, alpha2x, beta))
            : dy[i];
  }
} // HardSwishGrad

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void HardSwish<T, CUDAContext>(                                        \
      const int N,                                                       \
      const float alpha,                                                 \
      const float beta,                                                  \
      const T* x,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    _HardSwish<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                               \
        convert::To<math::AccmulatorType<T>::type>(alpha),               \
        convert::To<math::AccmulatorType<T>::type>(beta),                \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),           \
        reinterpret_cast<math::ScalarType<T>::type*>(y));                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void HardSwishGrad<T, CUDAContext>(                                        \
      const int N,                                                           \
      const float alpha,                                                     \
      const float beta,                                                      \
      const T* dy,                                                           \
      const T* x,                                                            \
      T* dx,                                                                 \
      CUDAContext* ctx) {                                                    \
    _HardSwishGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                                   \
        convert::To<math::AccmulatorType<T>::type>(alpha),                   \
        convert::To<math::AccmulatorType<T>::type>(beta),                    \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),              \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),               \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));                   \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
