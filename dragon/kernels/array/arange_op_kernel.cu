#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_Arange(const int nthreads, const float start, const float step, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = start + (float)i * step;
  }
}

template <>
__global__ void _Arange<half>(
    const int nthreads,
    const float start,
    const float step,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __float2half(start + (float)i * step);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Arange<float16, CUDAContext>(
    const int count,
    const float start,
    const float step,
    float16* y,
    CUDAContext* ctx) {
  _Arange<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, start, step, reinterpret_cast<half*>(y));
}

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void Arange<T, CUDAContext>(                                            \
      const int count,                                                    \
      const float start,                                                  \
      const float step,                                                   \
      T* y,                                                               \
      CUDAContext* ctx) {                                                 \
    _Arange<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, start, step, y);                                           \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
