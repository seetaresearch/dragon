#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _IndexSelect(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const int select_dim,
    const int64_t* index,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int j = yi % inner_dim;
    const int i = yi / inner_dim / select_dim;
#if __CUDA_ARCH__ >= 350
    int pos = __ldg(index + ((yi / inner_dim) % select_dim));
#else
    int pos = index[(yi / inner_dim) % select_dim];
#endif
    pos = pos >= 0 ? pos : pos + axis_dim;
    y[yi] = x[(i * axis_dim + pos) * inner_dim + j];
  }
}

template <typename T>
__global__ void _IndexSelectGrad(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const int select_dim,
    const int64_t* index,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(ti, nthreads) {
    const int i = ti / inner_dim;
    const int j = ti % inner_dim;
    const int x_offset = i * axis_dim * inner_dim + j;
    const T* offset_dy = dy + i * select_dim * inner_dim + j;
    for (int k = 0; k < select_dim; ++k) {
#if __CUDA_ARCH__ >= 350
      int pos = __ldg(index + k);
#else
      int pos = index[k];
#endif
      pos = pos >= 0 ? pos : pos + axis_dim;
      dx[x_offset + pos * inner_dim] += (*offset_dy);
      offset_dy += inner_dim;
    }
  }
}

template <>
__global__ void _IndexSelectGrad<half>(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const int select_dim,
    const int64_t* index,
    const half* dy,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(ti, nthreads) {
    const int i = ti / inner_dim;
    const int j = ti % inner_dim;
    const int x_offset = i * axis_dim * inner_dim + j;
    const half* offset_dy = dy + i * select_dim * inner_dim + j;
    for (int k = 0; k < select_dim; ++k) {
#if __CUDA_ARCH__ >= 350
      int pos = __ldg(index + k);
#else
      int pos = index[k];
#endif
      pos = pos >= 0 ? pos : pos + axis_dim;
      pos = x_offset + pos * inner_dim;
#if __CUDA_ARCH__ >= 530
      dx[pos] = __hadd(dx[pos], *(offset_dy));
#else
      dx[pos] =
          __float2half(__half2float(dx[pos]) + __half2float(*(offset_dy)));
#endif
      offset_dy += inner_dim;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void IndexSelectGrad<float16, CUDAContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int select_dim,
    const int64_t* index,
    const float16* dy,
    float16* dx,
    CUDAContext* ctx) {
  const int nthreads = outer_dim * inner_dim;
  _IndexSelectGrad<<<
      CUDA_BLOCKS(nthreads),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(
      nthreads,
      inner_dim,
      axis_dim,
      select_dim,
      index,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
} // IndexSelectGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                \
  template <>                                                    \
  void IndexSelect<T, CUDAContext>(                              \
      const int outer_dim,                                       \
      const int inner_dim,                                       \
      const int axis_dim,                                        \
      const int select_dim,                                      \
      const int64_t* index,                                      \
      const T* x,                                                \
      T* y,                                                      \
      CUDAContext* ctx) {                                        \
    const int nthreads = outer_dim * select_dim * inner_dim;     \
    _IndexSelect<<<                                              \
        CUDA_BLOCKS(nthreads),                                   \
        CUDA_THREADS,                                            \
        0,                                                       \
        ctx->cuda_stream()>>>(                                   \
        nthreads, inner_dim, axis_dim, select_dim, index, x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                             \
  template <>                                                      \
  void IndexSelectGrad<T, CUDAContext>(                            \
      const int outer_dim,                                         \
      const int inner_dim,                                         \
      const int axis_dim,                                          \
      const int select_dim,                                        \
      const int64_t* index,                                        \
      const T* dy,                                                 \
      T* dx,                                                       \
      CUDAContext* ctx) {                                          \
    const int nthreads = outer_dim * inner_dim;                    \
    _IndexSelectGrad<<<                                            \
        CUDA_BLOCKS(nthreads),                                     \
        CUDA_THREADS,                                              \
        0,                                                         \
        ctx->cuda_stream()>>>(                                     \
        nthreads, inner_dim, axis_dim, select_dim, index, dy, dx); \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
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
