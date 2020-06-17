#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Concat(
    const int nthreads,
    const int inner_dim,
    const int x_cols,
    const int y_cols,
    const int offset,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int i = xi / x_cols;
    const int j = xi % x_cols;
    y[i * y_cols + offset + j] = x[xi];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void Concat<T, CUDAContext>(                                               \
      const int outer_dim,                                                   \
      const int inner_dim,                                                   \
      const int x_axis_dim,                                                  \
      const int y_axis_dim,                                                  \
      const int index,                                                       \
      const T* x,                                                            \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    const int offset = index * inner_dim;                                    \
    const int x_cols = x_axis_dim * inner_dim;                               \
    const int y_cols = y_axis_dim * inner_dim;                               \
    const int nthreads = outer_dim * x_cols;                                 \
    _Concat<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, inner_dim, x_cols, y_cols, offset, x, y);                  \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
