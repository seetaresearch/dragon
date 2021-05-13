#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/conversions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_Clip(const int N, const AccT low, const AccT high, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<T>(max(low, min(convert::To<AccT>(x[i]), high)));
  }
}

template <typename T, typename AccT>
__global__ void _ClipGrad(
    const int N,
    const AccT low,
    const AccT high,
    const T* dy,
    const T* x,
    T* dx) {
  const T kZero = convert::To<T>(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    dx[i] = val < low || val > high ? kZero : dy[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                             \
  template <>                                                       \
  void Clip<T, CUDAContext>(                                        \
      const int N,                                                  \
      const float low,                                              \
      const float high,                                             \
      const T* x,                                                   \
      T* y,                                                         \
      CUDAContext* ctx) {                                           \
    _Clip<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, low, high, x, y);                                        \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                            \
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
        N, low, high, dy, x, dx);                                       \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
