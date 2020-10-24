#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_ReduceLossGrad(const int nthreads, const T scale, const T* dy, T* dx) {
#if __CUDA_ARCH__ >= 350
  const T alpha = __ldg(dy) * scale;
#else
  const T alpha = dy[0] * scale;
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] *= alpha;
  }
}

__global__ void _ReduceLossGrad(
    const int nthreads,
    const float scale,
    const half* dy,
    half* dx) {
#if __CUDA_ARCH__ >= 350
  const float alpha = __half2float(__ldg(dy)) * scale;
#else
  const float alpha = __half2float(dy[0]) * scale;
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] = __float2half(__half2float(dx[i]) * alpha);
  }
}

template <typename T>
__global__ void
_ReduceLossGrad(const int nthreads, const T* normalizer, const T* dy, T* dx) {
#if __CUDA_ARCH__ >= 350
  const T alpha = __ldg(dy) / max(__ldg(normalizer), T(1));
#else
  const T alpha = dy[0] / max(normalizer[0], T(1));
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] *= alpha;
  }
}

template <>
__global__ void _ReduceLossGrad<half>(
    const int nthreads,
    const half* normalizer,
    const half* dy,
    half* dx) {
#if __CUDA_ARCH__ >= 350
  const float alpha =
      __half2float(__ldg(dy)) / max(__half2float(__ldg(normalizer)), 1.f);
#else
  const float alpha =
      __half2float(dy[0]) / max(__half2float(normalizer[0]), 1.f);
#endif
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] = __float2half(__half2float(dx[i]) * alpha);
  }
}

template <typename T>
__global__ void _BroadcastLossGrad(
    const int nthreads,
    const int dim1,
    const int dim2,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] *= __ldg(dy + (i / dim1) * dim2 + (i % dim2));
#else
    dx[i] *= dy[(i / dim1) * dim2 + (i % dim2)];
#endif
  }
}

template <>
__global__ void _BroadcastLossGrad<half>(
    const int nthreads,
    const int dim1,
    const int dim2,
    const half* dy,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = __float2half(
        __half2float(dx[i]) *
        __half2float(__ldg(dy + (i / dim1) * dim2 + (i % dim2))));
#else
    dx[i] = __float2half(
        __half2float(dx[i]) * __half2float(dy[(i / dim1) * dim2 + (i % dim2)]));
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ReduceLossGrad<float16, CUDAContext>(
    const int count,
    const int num_masks,
    const float normalizer,
    const float16* dy,
    const float16* mask,
    float16* dx,
    CUDAContext* ctx) {
  if (num_masks > 0 && normalizer < 0.f) {
    auto* normalizer_v2 = const_cast<float16*>(mask + num_masks);
    math::Sum(num_masks, 1.f, mask, normalizer_v2, ctx);
    _ReduceLossGrad<<<
        CUDA_BLOCKS(count),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        count,
        reinterpret_cast<const half*>(normalizer_v2),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  } else {
    _ReduceLossGrad<<<
        CUDA_BLOCKS(count),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        count,
        1.f / std::max(0.5f, normalizer),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  }
}

template <>
void BroadcastLossGrad<float16, CUDAContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const float16* dy,
    float16* dx,
    CUDAContext* ctx) {
  const auto nthreads = outer_dim * axis_dim * inner_dim;
  _BroadcastLossGrad<<<
      CUDA_BLOCKS(nthreads),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(
      nthreads,
      axis_dim * inner_dim,
      inner_dim,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
}

#define DEFINE_KERNEL_LAUNCHER(T)                                   \
  template <>                                                       \
  void ReduceLoss<T, CUDAContext>(                                  \
      const int count,                                              \
      const int num_masks,                                          \
      const float normalizer,                                       \
      const T* x,                                                   \
      const T* mask,                                                \
      T* y,                                                         \
      CUDAContext* ctx) {                                           \
    if (num_masks > 0 && normalizer < 0.f) {                        \
      auto* normalizer_v2 = const_cast<T*>(mask + num_masks);       \
      math::Sum(num_masks, 1.f, mask, normalizer_v2, ctx);          \
      math::Sum(count, 1.f, x, y, ctx);                             \
      math::Div(1, y, normalizer_v2, y, ctx);                       \
    } else {                                                        \
      math::Sum(count, 1.f / std::max(1.f, normalizer), x, y, ctx); \
    }                                                               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                         \
  template <>                                                  \
  void ReduceLossGrad<T, CUDAContext>(                         \
      const int count,                                         \
      const int num_masks,                                     \
      const float normalizer,                                  \
      const T* dy,                                             \
      const T* mask,                                           \
      T* dx,                                                   \
      CUDAContext* ctx) {                                      \
    if (num_masks > 0 && normalizer < 0.f) {                   \
      auto* normalizer_v2 = const_cast<T*>(mask + num_masks);  \
      math::Sum(num_masks, 1.f, mask, normalizer_v2, ctx);     \
      _ReduceLossGrad<<<                                       \
          CUDA_BLOCKS(count),                                  \
          CUDA_THREADS,                                        \
          0,                                                   \
          ctx->cuda_stream()>>>(count, normalizer_v2, dy, dx); \
    } else {                                                   \
      _ReduceLossGrad<<<                                       \
          CUDA_BLOCKS(count),                                  \
          CUDA_THREADS,                                        \
          0,                                                   \
          ctx->cuda_stream()>>>(                               \
          count, T(1.f / std::max(0.5f, normalizer)), dy, dx); \
    }                                                          \
  }                                                            \
  template <>                                                  \
  void BroadcastLossGrad<T, CUDAContext>(                      \
      const int outer_dim,                                     \
      const int inner_dim,                                     \
      const int axis_dim,                                      \
      const T* dy,                                             \
      T* dx,                                                   \
      CUDAContext* ctx) {                                      \
    const auto nthreads = outer_dim * axis_dim * inner_dim;    \
    _BroadcastLossGrad<<<                                      \
        CUDA_BLOCKS(nthreads),                                 \
        CUDA_THREADS,                                          \
        0,                                                     \
        ctx->cuda_stream()>>>(                                 \
        nthreads, axis_dim * inner_dim, inner_dim, dy, dx);    \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
