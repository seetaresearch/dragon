#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Sigmoid <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Sigmoid(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = (T)1 / ((T)1 + exp(-x[idx]));
    }
}

template<> void Sigmoid<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Sigmoid<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x, y);
}

/*! SigmoidGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = dy[idx] * y[idx] * (1 - y[idx]);
    }
}

template<> void SigmoidGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _SigmoidGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dy, y, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA