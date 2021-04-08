#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, bool kUpper>
__global__ void _SetTriangular(
    const int nthreads,
    const int M,
    const int N,
    const int k,
    T* y) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int j = index % N;
    const int i = (index / N) % M;
    if (kUpper) {
      y[index] = j < i + k ? convert::To<T>(0.f) : y[index];
    } else {
      y[index] = j > i + k ? convert::To<T>(0.f) : y[index];
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void SetTriangular<T, CUDAContext>(                                       \
      const int batch_size,                                                 \
      const int M,                                                          \
      const int N,                                                          \
      const int k,                                                          \
      const int upper,                                                      \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    const auto nthreads = batch_size * M * N;                               \
    math::Copy(nthreads, x, y, ctx);                                        \
    if (upper > 0) {                                                        \
      _SetTriangular<T, true>                                               \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, k, y);                                        \
    } else {                                                                \
      _SetTriangular<T, false>                                              \
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              nthreads, M, N, k, y);                                        \
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
