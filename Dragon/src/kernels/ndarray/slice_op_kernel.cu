#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Slice <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Slice(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = y_slice_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int slice_idx = idx % tmp;
        const int x_idx = (outer_idx * x_slice_dim + slice_offset)
                                * inner_dim + slice_idx;
        y[idx] = x[x_idx];
    }
}

template <> void Slice<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Slice<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                slice_offset, x, y);
}

/*! SliceGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SliceGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = y_slice_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int slice_idx = idx % tmp;
        const int x_idx = (outer_idx * x_slice_dim + slice_offset) 
                                * inner_dim + slice_idx;
        dx[x_idx] = dy[idx];
    }
}

template <> void SliceGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _SliceGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                slice_offset, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA