#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Tile <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Tile(
    const int               count,
    const int               ex_inner_dim,
    const int               multiple,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int d = idx % ex_inner_dim;
        const int n = idx / ex_inner_dim / multiple;
        const int x_idx = n * ex_inner_dim + d;
        y[idx] = x[x_idx];
    }
}

template <> void Tile<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Tile<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, ex_inner_dim, multiple, x, y);
}

/*! TileGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _TileGrad(
    const int               count,
    const int               ex_inner_dim,
    const int               multiple,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T gradient = 0;
        const int offset = (idx / ex_inner_dim * multiple)
            * ex_inner_dim + idx % ex_inner_dim;
        for (int t = 0; t < multiple; t++)
            gradient += dy[offset + t * ex_inner_dim];
        dx[idx] = gradient;
    }
}

template <> void TileGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _TileGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, ex_inner_dim, multiple, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA