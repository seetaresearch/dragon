#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Tanh <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Tanh(
    const int               count,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        y[i] = tanh(x[i]);
    }
}

template<> void Tanh<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Tanh<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x, y);
}

/*! TanhGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _TanhGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
}

template<> void TanhGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _TanhGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dy, y, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA