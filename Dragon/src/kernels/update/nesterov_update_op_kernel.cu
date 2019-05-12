#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _NesterovUpdate(
    const int               nthreads,
    const T                 lr,
    const T                 momentum,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        T hi = h[i];
        T hi_new = h[i] = momentum * hi + lr * g[i];
        g[i] = (1 + momentum) * hi_new - momentum * hi;
    }
}

template <> void NesterovUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _NesterovUpdate
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count, lr, momentum, g, h
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA