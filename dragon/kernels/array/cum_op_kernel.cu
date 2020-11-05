#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _CumSum(
    const int rows,
    const int cols,
    const int inner_dim,
    const bool exclusive,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, rows) {
    int c = (i / inner_dim) * cols * inner_dim + (i % inner_dim);
    y[c] = exclusive ? T(0) : x[c];
    for (int j = 1; j < cols; ++j) {
      const int yi = c + inner_dim;
      y[yi] = math::PlusFunctor<T>()(y[c], x[exclusive ? c : yi]);
      c = yi;
    }
  }
}

template <>
__global__ void _CumSum<half>(
    const int rows,
    const int cols,
    const int inner_dim,
    const bool exclusive,
    const half* x,
    half* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, rows) {
    int c = (i / inner_dim) * cols * inner_dim + (i % inner_dim);
    y[c] = exclusive ? kZero : x[c];
    for (int j = 1; j < cols; ++j) {
      const int yi = c + inner_dim;
      y[yi] = math::PlusFunctor<half>()(y[c], x[exclusive ? c : yi]);
      c = yi;
    }
  }
}

template <typename T>
__global__ void _CumSumReverse(
    const int rows,
    const int cols,
    const int inner_dim,
    const bool exclusive,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, rows) {
    int c = ((i / inner_dim) * cols + (cols - 1)) * inner_dim + (i % inner_dim);
    y[c] = exclusive ? T(0) : x[c];
    for (int j = cols - 2; j >= 0; --j) {
      const int yi = c - inner_dim;
      y[yi] = math::PlusFunctor<T>()(y[c], x[exclusive ? c : yi]);
      c = yi;
    }
  }
}

template <>
__global__ void _CumSumReverse<half>(
    const int rows,
    const int cols,
    const int inner_dim,
    const bool exclusive,
    const half* x,
    half* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, rows) {
    int c = ((i / inner_dim) * cols + (cols - 1)) * inner_dim + (i % inner_dim);
    y[c] = exclusive ? kZero : x[c];
    for (int j = cols - 2; j >= 0; --j) {
      const int yi = c - inner_dim;
      y[yi] = math::PlusFunctor<half>()(y[c], x[exclusive ? c : yi]);
      c = yi;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void CumSum<float16, CUDAContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const bool exclusive,
    const bool reverse,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  auto rows = outer_dim * inner_dim, cols = axis_dim;
  if (reverse) {
    _CumSumReverse<<<CUDA_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        rows,
        cols,
        inner_dim,
        exclusive,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  } else {
    _CumSum<<<CUDA_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        rows,
        cols,
        inner_dim,
        exclusive,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void CumSum<T, CUDAContext>(                                             \
      const int outer_dim,                                                 \
      const int inner_dim,                                                 \
      const int axis_dim,                                                  \
      const bool exclusive,                                                \
      const bool reverse,                                                  \
      const T* x,                                                          \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    auto rows = outer_dim * inner_dim, cols = axis_dim;                    \
    if (reverse) {                                                         \
      _CumSumReverse<<<                                                    \
          CUDA_BLOCKS(rows),                                               \
          CUDA_THREADS,                                                    \
          0,                                                               \
          ctx->cuda_stream()>>>(rows, cols, inner_dim, exclusive, x, y);   \
    } else {                                                               \
      _CumSum<<<CUDA_BLOCKS(rows), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          rows, cols, inner_dim, exclusive, x, y);                         \
    }                                                                      \
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
