#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/cast.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _ApplyMask(
    const int nthreads,
    const T scale,
    const T* x,
    const uint8_t* mask,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = x[i] * (T)mask[i] * scale;
  }
}

__global__ void _ApplyMask(
    const int nthreads,
    const float scale,
    const half* x,
    const uint8_t* mask,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __float2half(__half2float(x[i]) * (float)mask[i] * scale);
  }
}

template <typename T>
__global__ void _Dropout(
    const int nthreads,
    const uint32_t threshold,
    const T scale,
    const T* x,
    const uint32_t* r,
    uint8_t* mask,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = x[i] * T(mask[i] = (r[i] > threshold)) * scale;
  }
}

__global__ void _Dropout(
    const int nthreads,
    const uint32_t threshold,
    const float scale,
    const half* x,
    const uint32_t* r,
    uint8_t* mask,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __float2half(
        __half2float(x[i]) * float(mask[i] = (r[i] > threshold)) * scale);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ApplyMask<float16, CUDAContext>(
    const int count,
    const float scale,
    const float16* x,
    const uint8_t* mask,
    float16* y,
    CUDAContext* ctx) {
  _ApplyMask<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      scale,
      reinterpret_cast<const half*>(x),
      mask,
      reinterpret_cast<half*>(y));
}

template <>
void Dropout<float16, CUDAContext>(
    const int count,
    const float ratio,
    const float scale,
    const float16* x,
    uint8_t* mask,
    float16* y,
    uint32_t* r,
    CUDAContext* ctx) {
  math::Random(count, r, ctx);
  _Dropout<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      static_cast<uint32_t>(UINT_MAX * ratio),
      scale,
      reinterpret_cast<const half*>(x),
      r,
      mask,
      reinterpret_cast<half*>(y));
}

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void ApplyMask<T, CUDAContext>(                                            \
      const int count,                                                       \
      const float scale,                                                     \
      const T* x,                                                            \
      const uint8_t* mask,                                                   \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    _ApplyMask<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, cast::to<T>(scale), x, mask, y);                              \
  }                                                                          \
  template <>                                                                \
  void Dropout<T, CUDAContext>(                                              \
      const int count,                                                       \
      const float ratio,                                                     \
      const float scale,                                                     \
      const T* x,                                                            \
      uint8_t* mask,                                                         \
      T* y,                                                                  \
      uint32_t* r,                                                           \
      CUDAContext* ctx) {                                                    \
    math::Random(count, r, ctx);                                             \
    auto threshold = static_cast<uint32_t>(UINT_MAX * ratio);                \
    _Dropout<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        count, threshold, cast::to<T>(scale), x, r, mask, y);                \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
