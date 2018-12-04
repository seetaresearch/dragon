#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SElu <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SElu(
    const int               count,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = x[idx] > 0 ? 1.0507f * x[idx] :
            1.7581f * (exp(x[idx]) - 1);
    }
}

template<> void SElu<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _SElu<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x, y);
}

/*! SEluGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SEluGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = y[idx] > 0 ? 1.0507f * dy[idx] :
            (1.7581f + y[idx]) * dy[idx];
    }
}

template<> void SEluGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _SEluGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dy, y, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA