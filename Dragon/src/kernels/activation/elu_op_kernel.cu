#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Elu(
    const int               count,
    const T*                x,
    const float             alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = x[idx] > 0 ? x[idx] :
            alpha * (exp(x[idx]) - 1);
    }
}

template<> void Elu<float, CUDAContext>(
    const int               count,
    const float             alpha,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Elu
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, x, alpha, y
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _EluGrad(
    const int               count,
    const float             alpha,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = dy[idx] * (
            (y[idx] > 0) + (alpha + y[idx]) * (y[idx] <= 0)
        );
    }
}

template<> void EluGrad<float, CUDAContext>(
    const int               count,
    const float             alpha,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _EluGrad
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, alpha, dy, y, dx
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA