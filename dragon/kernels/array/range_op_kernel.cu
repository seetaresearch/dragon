#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_Range(const int nthreads, const double start, const double delta, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = T(start + double(i) * delta);
  }
}

template <>
__global__ void _Range<half>(
    const int nthreads,
    const double start,
    const double delta,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __float2half(float(start + double(i) * delta));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Range<float16, CUDAContext>(
    const int count,
    const double start,
    const double delta,
    float16* y,
    CUDAContext* ctx) {
  _Range<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, start, delta, reinterpret_cast<half*>(y));
}

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void Range<T, CUDAContext>(                                            \
      const int count,                                                   \
      const double start,                                                \
      const double delta,                                                \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    _Range<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, start, delta, y);                                         \
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
