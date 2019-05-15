#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Sigmoid(
    const int               nthreads,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = T(1) / (T(1) + exp(-x[i]));
    }
}

template<> void Sigmoid<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Sigmoid
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, x, y
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidGrad(
    const int               nthreads,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        dx[i] = dy[i] * y[i] * (1 - y[i]);
    }
}

template<> void SigmoidGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _SigmoidGrad
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, dy, y, dx
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA