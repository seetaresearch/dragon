#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_BiasAdd(const int nthreads, const int axis_dim, const T* x, const T* b, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = math::PlusFunctor<T>()(x[i], __ldg(b + i % axis_dim));
#else
    y[i] = math::PlusFunctor<T>()(x[i], b[i % axis_dim]);
#endif
  }
}

template <typename T>
__global__ void _BiasAdd(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    const T* b,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = math::PlusFunctor<T>()(x[i], __ldg(b + (i / inner_dim) % axis_dim));
#else
    y[i] = math::PlusFunctor<T>()(x[i], b[(i / inner_dim) % axis_dim]);
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void BiasAdd<float16, CUDAContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const float16* x,
    const float16* b,
    float16* y,
    CUDAContext* ctx) {
  const auto nthreads = outer_dim * axis_dim * inner_dim;
  if (inner_dim == 1) {
    _BiasAdd<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        nthreads,
        axis_dim,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(b),
        reinterpret_cast<half*>(y));
  } else {
    _BiasAdd<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        nthreads,
        inner_dim,
        axis_dim,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(b),
        reinterpret_cast<half*>(y));
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void BiasAdd<T, CUDAContext>(                                          \
      const int outer_dim,                                               \
      const int inner_dim,                                               \
      const int axis_dim,                                                \
      const T* x,                                                        \
      const T* b,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    const auto nthreads = outer_dim * axis_dim * inner_dim;              \
    if (inner_dim == 1) {                                                \
      _BiasAdd<<<                                                        \
          CUDA_BLOCKS(nthreads),                                         \
          CUDA_THREADS,                                                  \
          0,                                                             \
          ctx->cuda_stream()>>>(nthreads, axis_dim, x, b, y);            \
    } else {                                                             \
      _BiasAdd<<<                                                        \
          CUDA_BLOCKS(nthreads),                                         \
          CUDA_THREADS,                                                  \
          0,                                                             \
          ctx->cuda_stream()>>>(nthreads, inner_dim, axis_dim, x, b, y); \
    }                                                                    \
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
