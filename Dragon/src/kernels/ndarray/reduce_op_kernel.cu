#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Sum <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Sum(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T*                x,
    float*                  y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T sum_val = (T)0;
        const int offset = (idx / inner_dim * axis_dim)
            * inner_dim + idx % inner_dim;
        for (int j = 0; j < axis_dim; j++)
            sum_val += x[offset + j * inner_dim];
        y[idx] = sum_val;
   }
}

template<> void Sum<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Sum<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, axis_dim, inner_dim, x, y);
}

/*! SumGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SumGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 coeff,
    const T*                dy,
    float*                  dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int offset = (idx / inner_dim * axis_dim)
            * inner_dim + idx % inner_dim;
        for (int j = 0; j < axis_dim; j++)
            dx[offset + j * inner_dim] = dy[idx] * coeff;
    }
}

template<> void SumGrad<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             coeff,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _SumGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, axis_dim, inner_dim, coeff, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA