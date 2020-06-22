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

template <>
__global__ void _ApplyMask<half>(
    const int nthreads,
    const half scale,
    const half* x,
    const uint8_t* mask,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __hmul(__hmul(x[i], scale), __float2half((float)mask[i]));
#endif
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
    y[i] = x[i] * (T)(mask[i] = (r[i] > threshold)) * scale;
  }
}

template <>
__global__ void _Dropout<half>(
    const int nthreads,
    const uint32_t threshold,
    const half scale,
    const half* x,
    const uint32_t* r,
    uint8_t* mask,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __hmul(
        __hmul(x[i], scale),
        __float2half((float)(mask[i] = (r[i] > threshold))));
#endif
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
      cast::to<half>(scale),
      reinterpret_cast<const half*>(x),
      mask,
      reinterpret_cast<half*>(y));
}

template <>
void Dropout<float16, CUDAContext>(
    const int count,
    const float prob,
    const float scale,
    const float16* x,
    uint8_t* mask,
    float16* y,
    uint32_t* scratch,
    CUDAContext* ctx) {
  math::RandomUniform(count, 0.f, 1.f, scratch, ctx);
  _Dropout<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      static_cast<uint32_t>(UINT_MAX * prob),
      cast::to<half>(scale),
      reinterpret_cast<const half*>(x),
      scratch,
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
      const float prob,                                                      \
      const float scale,                                                     \
      const T* x,                                                            \
      uint8_t* mask,                                                         \
      T* y,                                                                  \
      uint32_t* scratch,                                                     \
      CUDAContext* ctx) {                                                    \
    math::RandomUniform(count, 0.f, 1.f, scratch, ctx);                      \
    auto threshold = static_cast<uint32_t>(UINT_MAX * prob);                 \
    _Dropout<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        count, threshold, cast::to<T>(scale), x, scratch, mask, y);          \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
