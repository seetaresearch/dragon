#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _ChannelShuffle(
    const int nthreads,
    const int inner_dim,
    const int G,
    const int K,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int j = yi % inner_dim;
    const int gi = (yi / inner_dim) % G;
    const int ki = (yi / inner_dim / G) % K;
    const int i = yi / inner_dim / G / K;
    y[yi] = x[((i * G + gi) * K + ki) * inner_dim + j];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                            \
  template <>                                                \
  void ChannelShuffle<T, CUDAContext>(                       \
      const int outer_dim,                                   \
      const int inner_dim,                                   \
      const int axis_dim,                                    \
      const int group,                                       \
      const T* x,                                            \
      T* y,                                                  \
      CUDAContext* ctx) {                                    \
    auto nthreads = outer_dim * axis_dim * inner_dim;        \
    _ChannelShuffle<<<                                       \
        CUDA_BLOCKS(nthreads),                               \
        CUDA_THREADS,                                        \
        0,                                                   \
        ctx->cuda_stream()>>>(                               \
        nthreads, inner_dim, group, axis_dim / group, x, y); \
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
