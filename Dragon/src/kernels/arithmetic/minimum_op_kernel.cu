#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! MinimumE <T = float32, Device = CUDA> */

template <typename T>
__global__ void _MinimumE(
    const int               count,
    const T*                x1,
    const T*                x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = min(x1[idx], x2[idx]);
    }
}

template <> void MinimumE<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    float*                  y,
    CUDAContext*            ctx) {
    _MinimumE<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x1, x2, y);
}

/*! MinimumB <T = float32, Device = CUDA> */

template <typename T>
__global__ void _MinimumB(
    const int               count,
    const T*                x1,
    const T                 x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = min(x1[idx], x2);
    }
}

template <> void MinimumB<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    float*                  y,
    CUDAContext*            ctx) {
    _MinimumB<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x1, x2, y);
}

/*! MinimumEGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _MinimumEGrad(
    const int               count,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const bool dy_to_dx1 = x1[idx] < x2[idx];
        dx1[idx] = dy_to_dx1 ? dy[idx] : 0;
        dx2[idx] = dy_to_dx1 ? 0 : dy[idx];
    }
}

template <> void MinimumEGrad<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    const float*            dy,
    float*                  dx1,
    float*                  dx2,
    CUDAContext*            ctx) {
    _MinimumEGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x1, x2, dy, dx1, dx2);
}

/*! MinimumBGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _MinimumBGrad(
    const int               count,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx1[idx] = (x1[idx] < x2) ? dy[idx] : 0;
    }
}

template <> void MinimumBGrad<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    const float*            dy,
    float*                  dx1,
 /* float*                  dx2, */
    CUDAContext*            ctx) {
    _MinimumBGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x1, x2, dy, dx1);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA