#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, bool kUpper>
__global__ void
_SetEye(const int nthreads, const int M, const int N, const int k, T* y) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int i = index % M;
    if (kUpper) {
      const int j = i + k;
      y[index * N + min(j, N - 1)] =
          j < N ? convert::To<T>(1.f) : convert::To<T>(0.f);
    } else {
      const int j = i - k;
      y[index * N + max(j, 0)] =
          j < 0 ? convert::To<T>(0.f) : convert::To<T>(1.f);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void SetEye<T, CUDAContext>(                                              \
      const int batch_size,                                                 \
      const int M,                                                          \
      const int N,                                                          \
      const int k,                                                          \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    const auto nthreads = batch_size * M;                                   \
    math::Set(nthreads* N, convert::To<T>(0.f), y, ctx);                    \
    if (k > 0) {                                                            \
      _SetEye<T, true>                                                      \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, k, y);                                        \
    } else {                                                                \
      _SetEye<T, false>                                                     \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, -k, y);                                       \
    }                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
