#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SmoothL1 <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SmoothL1(
    const int               count,
    const float             beta,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const T val = x[idx];
        const T abs_val = abs(val);
        if (abs_val < beta) y[idx] = 0.5 * val * val / beta;
        else y[idx] = abs_val - 0.5 * beta;
    }
}

template<> void SmoothL1<float, CUDAContext>(
    const int               count,
    const float             beta,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _SmoothL1<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, beta, x, y);
}

/*! SmoothL1Grad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SmoothL1Grad(
    const int               count,
    const float             beta,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const T val = dy[idx];
        const T abs_val = abs(val);
        if (abs_val < beta) dx[idx] = val / beta;
        //  val > 0: 1 | val == 0: 0 | val < 0: -1
        else dx[idx] = (val > T(0)) - (val < T(0));
    }
}

template<> void SmoothL1Grad<float, CUDAContext>(
    const int               count,
    const float             beta,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _SmoothL1Grad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, beta, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA