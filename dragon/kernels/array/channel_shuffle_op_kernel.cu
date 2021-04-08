#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _ChannelShuffle(
    const int NxCxS,
    const int S,
    const int G,
    const int K,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(index, NxCxS) {
    const int j = index % S;
    const int gi = index / S % G;
    const int ki = index / S / G % K;
    const int i = index / S / G / K;
    y[index] = x[((i * G + gi) * K + ki) * S + j];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                        \
  template <>                                            \
  void ChannelShuffle<T, CUDAContext>(                   \
      const int N,                                       \
      const int S,                                       \
      const int C,                                       \
      const int G,                                       \
      const T* x,                                        \
      T* y,                                              \
      CUDAContext* ctx) {                                \
    const auto NxCxS = N * C * S;                        \
    _ChannelShuffle<<<                                   \
        CUDA_BLOCKS(NxCxS),                              \
        CUDA_THREADS,                                    \
        0,                                               \
        ctx->cuda_stream()>>>(NxCxS, S, G, C / G, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
