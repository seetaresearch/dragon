#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Repeat(
    const int nthreads,
    const int axis_dim,
    const int x_inner_dim,
    const int y_inner_dim,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int k = yi % x_inner_dim;
    const int j = (yi / y_inner_dim) % axis_dim;
    const int i = yi / y_inner_dim / axis_dim;
    const int xi = (i * axis_dim + j) * x_inner_dim + k;
    y[yi] = x[xi];
  }
}

template <typename T>
__global__ void _RepeatGrad(
    const int nthreads,
    const int axis_dim,
    const int x_inner_dim,
    const int y_inner_dim,
    const int repeats,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int k = xi % x_inner_dim;
    const int j = (xi / x_inner_dim) % axis_dim;
    const int i = xi / x_inner_dim / axis_dim;
    const T* offset_dy = dy + ((i * axis_dim + j) * y_inner_dim + k);
    T val = T(0);
    for (int r = 0; r < repeats; ++r)
      val += offset_dy[r * x_inner_dim];
    dx[xi] = val;
  }
}

template <>
__global__ void _RepeatGrad<half>(
    const int nthreads,
    const int axis_dim,
    const int x_inner_dim,
    const int y_inner_dim,
    const int repeats,
    const half* dy,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
#if __CUDA_ARCH__ >= 530
    const int k = xi % x_inner_dim;
    const int j = (xi / x_inner_dim) % axis_dim;
    const int i = xi / x_inner_dim / axis_dim;
    const half* offset_dy = dy + ((i * axis_dim + j) * y_inner_dim + k);
    float val = 0.f;
    for (int r = 0; r < repeats; ++r)
      val += __half2float(offset_dy[r * x_inner_dim]);
    dx[xi] = __float2half(val);
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void RepeatGrad<float16, CUDAContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int repeats,
    const float16* dy,
    float16* dx,
    CUDAContext* ctx) {
  auto y_inner_dim = inner_dim * repeats;
  auto nthreads = outer_dim * axis_dim * inner_dim;
  _RepeatGrad<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      nthreads,
      axis_dim,
      inner_dim,
      y_inner_dim,
      repeats,
      reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
} // RepeatGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void Repeat<T, CUDAContext>(                                               \
      const int outer_dim,                                                   \
      const int inner_dim,                                                   \
      const int axis_dim,                                                    \
      const int repeats,                                                     \
      const T* x,                                                            \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    auto y_inner_dim = inner_dim * repeats;                                  \
    auto nthreads = outer_dim * axis_dim * y_inner_dim;                      \
    _Repeat<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, axis_dim, inner_dim, y_inner_dim, x, y);                   \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void RepeatGrad<T, CUDAContext>(                                    \
      const int outer_dim,                                            \
      const int inner_dim,                                            \
      const int axis_dim,                                             \
      const int repeats,                                              \
      const T* dy,                                                    \
      T* dx,                                                          \
      CUDAContext* ctx) {                                             \
    auto y_inner_dim = inner_dim * repeats;                           \
    auto nthreads = outer_dim * axis_dim * inner_dim;                 \
    _RepeatGrad<<<                                                    \
        CUDA_BLOCKS(nthreads),                                        \
        CUDA_THREADS,                                                 \
        0,                                                            \
        ctx->cuda_stream()>>>(                                        \
        nthreads, axis_dim, inner_dim, y_inner_dim, repeats, dy, dx); \
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
