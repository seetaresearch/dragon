#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_HardSigmoid(const int N, const AccT alpha, const AccT beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT s_val = fma(convert::To<AccT>(x[i]), alpha, beta);
    y[i] = convert::To<T>(max(AccT(0), min(AccT(1), s_val)));
  }
}

template <typename T, typename AccT>
__global__ void _HardSigmoidGrad(
    const int N,
    const AccT alpha,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(y[i]);
    dx[i] = convert::To<T>(
        (val > AccT(0) && val < AccT(1)) ? convert::To<AccT>(dy[i]) * alpha
                                         : AccT(0));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

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
        convert::To<math::AccmulatorType<T>::type>(alpha),                 \
        convert::To<math::AccmulatorType<T>::type>(beta),                  \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),             \
        reinterpret_cast<math::ScalarType<T>::type*>(y));                  \
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
        convert::To<math::AccmulatorType<T>::type>(alpha),                     \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),                \
        reinterpret_cast<const math::ScalarType<T>::type*>(y),                 \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));                     \
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
