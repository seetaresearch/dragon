#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Tanh(
    const int               nthreads,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = tanh(x[i]);
    }
}

template<> void Tanh<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Tanh
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, x, y
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _TanhGrad(
    const int               nthreads,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
}

template<> void TanhGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _TanhGrad
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, dy, y, dx
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA