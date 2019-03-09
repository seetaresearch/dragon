#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! RMSPropUpdate <T = float32, Device = CUDA> */

template <typename T>
__global__ void _RMSPropUpdate(
    const int               count,
    const T                 lr,
    const T                 decay,
    const T                 eps,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T gi = g[i];
        T hi = h[i] = decay * h[i] + (1 - decay) * gi * gi;
        g[i] = lr * g[i] / (sqrt(hi) + eps);
    }
}

template <> void RMSPropUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _RMSPropUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, lr, decay, eps, g, h);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA