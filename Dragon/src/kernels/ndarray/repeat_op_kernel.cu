#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Repeat <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Repeat(
    const int               count,
    const int               inner_dim,
    const int               repeats,
    const int               dim,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int d = idx % inner_dim;
        const int b = (idx / inner_dim / repeats) % dim;
        const int n = idx / inner_dim / repeats / dim;
        const int x_idx = (n * dim + b) * inner_dim + d;
        y[idx] = x[x_idx];
    }
}

template <> void Repeat<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Repeat<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, inner_dim, repeats, dim, x, y);
}

/*! RepeatGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _RepeatGrad(
    const int               count,
    const int               inner_dim,
    const int               repeats,
    const int               dim,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int d = idx % inner_dim;
        const int b = (idx / inner_dim) % dim;
        const int n = idx / inner_dim  / dim;
        T gradient = 0;
        for (int t = 0; t < repeats; t++)
            gradient += dy[
                (((n * dim + b) * repeats) + t)
                    * inner_dim + d];
        dx[idx] = gradient;
    }
}

template <> void RepeatGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _RepeatGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, inner_dim, repeats, dim, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA