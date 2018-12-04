#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Clip <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Clip(
    const int               count,
    const T                 low,
    const T                 high,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = max(low, min(x[idx], high));
    }
}

template <> void Clip<float, CUDAContext>(
    const int               count,
    const float             low,
    const float             high,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Clip<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, low, high, x, y);
}

/*! ClipGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ClipGrad(
    const int               count,
    const T                 low,
    const T                 high,
    const T*                x,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const T xi = x[idx];
        dx[idx] = (xi < low || xi > high) ? 0 : dy[idx];
    }
}

template <> void ClipGrad<float, CUDAContext>(
    const int               count,
    const float             low,
    const float             high,
    const float*            x,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ClipGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, low, high, x, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA