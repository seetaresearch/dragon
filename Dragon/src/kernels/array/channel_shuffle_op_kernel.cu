#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _ChannelShuffle(
    const int               nthreads,
    const int               inner_dim,
    const int               G, 
    const int               K,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int inner_idx = yi % inner_dim;
        const int gi = (yi / inner_dim) % G;
        const int ki = (yi / inner_dim / G) % K;
        const int outer_idx = yi / inner_dim / G / K;
        y[yi] = x[((outer_idx * G + gi) * K + ki
                        ) * inner_dim + inner_idx];
   }
}

/* Kernel Launchers */

#define DEFINE_SHUFFLE_KERNEL_LAUNCHER(T) \
    template <> void ChannelShuffle<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               group, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto nthreads = outer_dim * axis_dim * inner_dim; \
        _ChannelShuffle \
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            nthreads, \
            inner_dim, \
            group, \
            axis_dim / group, \
            x, y \
        ); \
    }

DEFINE_SHUFFLE_KERNEL_LAUNCHER(bool);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(int8_t);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(uint8_t);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(int);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(int64_t);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(float16);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(float);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(double);

#undef DEFINE_SHUFFLE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA