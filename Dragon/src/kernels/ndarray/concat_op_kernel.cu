#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Concat <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Concat(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = x_concat_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int concat_idx = idx % tmp;
        const int y_idx = (outer_idx * y_concat_dim + concat_offset)
                                     * inner_dim + concat_idx;
        y[y_idx] = x[idx];
    }
}

template <> void Concat<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Concat<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_concat_dim, y_concat_dim,
                concat_offset, x, y);
}

/*! Concat <T = float16, Device = CUDA> */

template <typename T>
__global__ void _ConcatHalf(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = x_concat_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int concat_idx = idx % tmp;
        const int y_idx = (outer_idx * y_concat_dim + concat_offset)
                                * inner_dim + concat_idx;
        y[y_idx] = x[idx];
    }
}

template <> void Concat<float16, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _ConcatHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_concat_dim, y_concat_dim, concat_offset,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
}

/*! ConcatGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ConcatGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = x_concat_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int concat_idx = idx % tmp;
        const int y_idx = (outer_idx * y_concat_dim + concat_offset)
                                     * inner_dim + concat_idx;
        dx[idx] = dy[y_idx];
    }
}

template <> void ConcatGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ConcatGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_concat_dim, y_concat_dim,
                concat_offset, dy, dx);
}

/*! ConcatGrad <T = float16, Device = CUDA> */

template <typename T>
__global__ void _ConcatGradHalf(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = x_concat_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int concat_idx = idx % tmp;
        const int y_idx = (outer_idx * y_concat_dim + concat_offset)
                                * inner_dim + concat_idx;
        dx[idx] = dy[y_idx];
    }
}

template <> void ConcatGrad<float16, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    _ConcatGradHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, outer_dim, inner_dim,
            x_concat_dim, y_concat_dim, concat_offset,
                reinterpret_cast<const half*>(dy),
                    reinterpret_cast<half*>(dx));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA