#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Affine(
    const int nthreads,
    const int axis_dim,
    const int inner_dim,
    const T* x,
    const T* w,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(w + (i / inner_dim) % axis_dim) * x[i];
#else
    y[i] = w[(i / inner_dim) % axis_dim] * x[i];
#endif
  }
}

template <>
__global__ void _Affine<half>(
    const int nthreads,
    const int axis_dim,
    const int inner_dim,
    const half* x,
    const half* w,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __hmul(x[i], __ldg(w + (i / inner_dim) % axis_dim));
#endif
  }
}

template <typename T>
__global__ void _Affine(
    const int nthreads,
    const int axis_dim,
    const int inner_dim,
    const T* x,
    const T* w,
    const T* b,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int wi = (i / inner_dim) % axis_dim;
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(w + wi) * x[i] + __ldg(b + wi);
#else
    y[i] = w[wi] * x[i] + b[wi];
#endif
  }
}

template <>
__global__ void _Affine<half>(
    const int nthreads,
    const int axis_dim,
    const int inner_dim,
    const half* x,
    const half* w,
    const half* b,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    const int wi = (i / inner_dim) % axis_dim;
    y[i] = __hadd(__hmul(x[i], __ldg(w + wi)), __ldg(b + wi));
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Affine<float16, CUDAContext>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float16* x,
    const float16* w,
    const float16* b,
    float16* y,
    CUDAContext* ctx) {
  const int nthreads = outer_dim * axis_dim * inner_dim;
  if (b != nullptr) {
    _Affine<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        nthreads,
        axis_dim,
        inner_dim,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(w),
        reinterpret_cast<const half*>(b),
        reinterpret_cast<half*>(y));
  } else {
    _Affine<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        nthreads,
        axis_dim,
        inner_dim,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(w),
        reinterpret_cast<half*>(y));
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void Affine<T, CUDAContext>(                                                 \
      const int outer_dim,                                                     \
      const int axis_dim,                                                      \
      const int inner_dim,                                                     \
      const T* x,                                                              \
      const T* w,                                                              \
      const T* b,                                                              \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    const int nthreads = outer_dim * axis_dim * inner_dim;                     \
    if (b != nullptr) {                                                        \
      _Affine<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          nthreads, axis_dim, inner_dim, x, w, b, y);                          \
    } else {                                                                   \
      _Affine<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          nthreads, axis_dim, inner_dim, x, w, y);                             \
    }                                                                          \
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
