#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! CanonicalAxis <T = int32, Device = CUDA> */

template <typename T>
__global__ void _CanonicalAxis(
    const int               count,
    const int               dim,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (y[idx] < 0) y[idx] += dim;
    }
}

template <> void CanonicalAxis<int, CUDAContext>(
    const int               count,
    const int               dim,
    int*                    y,
    CUDAContext*            ctx) {
    _CanonicalAxis<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, y);
}

/*! Gather <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Gather(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int outer_idx = idx / inner_dim / y_slice_dim;
        const int slice_idx = idx % inner_dim;
        const int y_idx_offset = (idx / inner_dim) % y_slice_dim;
        const int x_idx_offset = indices[y_idx_offset];
        const int x_idx = (outer_idx * x_slice_dim + x_idx_offset)
                                     * inner_dim + slice_idx;
        y[idx] = x[x_idx];
    }
}

/*! Gather <T = float32, Device = CUDA> */

template <> void Gather<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Gather<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                indices, x, y);
}

/*! Gather <T = int32, Device = CUDA> */

template <> void Gather<int, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              x,
    int*                    y,
    CUDAContext*            ctx) {
    _Gather<int>
        << <CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                indices, x, y);
}

/*! GatherGrad <T = ?, Device = CUDA> */

template <typename T>
__global__ void _GatherGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int outer_idx = idx / inner_dim / y_slice_dim;
        const int slice_idx = idx % inner_dim;
        const int y_idx_offset = (idx / inner_dim) % y_slice_dim;
        const int x_idx_offset = indices[y_idx_offset];
        const int x_idx = (outer_idx * x_slice_dim + x_idx_offset)
                                     * inner_dim + slice_idx;
        atomicAdd(dx + x_idx, dy[idx]);
    }
}

/*! GatherGrad <T = float32, Device = CUDA> */

template <> void GatherGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _GatherGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                indices, dy, dx);
}

/*! GatherGrad <T = int32, Device = CUDA> */

template <> void GatherGrad<int, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              dy,
    int*                    dx,
    CUDAContext*            ctx) {
    _GatherGrad<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                indices, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA