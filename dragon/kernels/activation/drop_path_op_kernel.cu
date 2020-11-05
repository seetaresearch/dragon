#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/cast.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _DropPath(
    const int nthreads,
    const int cols,
    const float thresh,
    const T scale,
    const T* x,
    const float* mask,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = x[i] * T(__ldg(mask + (i / cols)) > thresh) * scale;
#else
    y[i] = x[i] * T(mask[i / cols] > thresh) * scale;
#endif
  }
}

__global__ void _DropPath(
    const int nthreads,
    const int cols,
    const float thresh,
    const float scale,
    const half* x,
    const float* mask,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __float2half(
        __half2float(x[i]) * float(__ldg(mask + (i / cols)) > thresh) * scale);
#else
    y[i] = __float2half(
        __half2float(x[i]) * float(mask[i / cols] > thresh) * scale);
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void DropPath<float16, CUDAContext>(
    const int rows,
    const int cols,
    const float scale,
    const float16* x,
    const float* mask,
    float16* y,
    CUDAContext* ctx) {
  const auto nthreads = rows * cols;
  const auto thresh = 1.f - (1.f / scale);
  _DropPath<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      nthreads,
      cols,
      thresh,
      scale,
      reinterpret_cast<const half*>(x),
      mask,
      reinterpret_cast<half*>(y));
}

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void DropPath<T, CUDAContext>(                                               \
      const int rows,                                                          \
      const int cols,                                                          \
      const float scale,                                                       \
      const T* x,                                                              \
      const float* mask,                                                       \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    const auto nthreads = rows * cols;                                         \
    const auto thresh = 1.f - (1.f / scale);                                   \
    _DropPath<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, cols, thresh, cast::to<T>(scale), x, mask, y);               \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
