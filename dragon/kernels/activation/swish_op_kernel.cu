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
__global__ void _Swish(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = LDG(x, i) / (T(1) + exp(-LDG(x, i)));
  }
}

template <>
__global__ void _Swish<half>(const int N, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(LDG2(x, i) / (1.f + exp(-LDG2(x, i))));
  }
}

template <typename T>
__global__ void
_SwishGrad(const int N, const T* dy, const T* x, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (LDG(y, i) + (T(1) - LDG(y, i)) / (T(1) + exp(-x[i])));
  }
}

template <>
__global__ void _SwishGrad<half>(
    const int N,
    const half* dy,
    const half* x,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __float2half(
        __half2float(dy[i]) *
        (LDG2(y, i) + (1.f - LDG2(y, i)) / (1.f + exp(-__half2float(x[i])))));
  }
} // SwishGrad

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                    \
  template <>                                                        \
  void Swish<T, CUDAContext>(                                        \
      const int N, const T* x, T* y, CUDAContext* ctx) {             \
    _Swish<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                           \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),       \
        reinterpret_cast<math::ScalarType<T>::type*>(y));            \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                   \
  template <>                                                            \
  void SwishGrad<T, CUDAContext>(                                        \
      const int N,                                                       \
      const T* dy,                                                       \
      const T* x,                                                        \
      const T* y,                                                        \
      T* dx,                                                             \
      CUDAContext* ctx) {                                                \
    _SwishGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                               \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),          \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),           \
        reinterpret_cast<const math::ScalarType<T>::type*>(y),           \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));               \
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
