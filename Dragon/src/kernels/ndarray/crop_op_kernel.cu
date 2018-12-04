#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Crop1d <T = ?, Device = CUDA> */

template<typename T>
__global__ void _Crop1d(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        y[idx] = x[(o * dim + ex_d + start) * inner_dim + i];
    }
}

/*! Crop1d <T = float32, Device = CUDA> */

template<> void Crop1d<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Crop1d<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, start, x, y);
}

/*! Crop1d <T = int32, Device = CUDA> */

template<> void Crop1d<int, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int*              x,
    int*                    y,
    CUDAContext*            ctx) {
    _Crop1d<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, start, x, y);
}

/*! Crop1dGrad <T = ?, Device = CUDA> */

template<typename T>
__global__ void _Crop1dGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int d = (idx / inner_dim) % dim;
        const int o = idx / inner_dim / dim;
        dx[idx] = (d < start || d >= end) ? 0 :
            dy[(o * ex_dim + d - start) * inner_dim + i];
    }
}

/*! Crop1dGrad <T = float32, Device = CUDA> */

template<> void Crop1dGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _Crop1dGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, start, end, dy, dx);
}

/*! Crop1dGrad <T = int32, Device = CUDA> */

template<> void Crop1dGrad<int, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const int*              dy,
    int*                    dx,
    CUDAContext*            ctx) {
    _Crop1dGrad<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, start, end, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA