  #ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SoftmaxCrossEntropy(
    const int               nthreads,
    const T*                prob,
    const T*                targets,
    T*                      losses) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        losses[i] = -targets[i] * log(
            max(prob[i], FLT_MIN)
        );
    }
}

template <> void SoftmaxCrossEntropy<float, CUDAContext>(
    const int               count,
    const float*            prob,
    const float*            targets,
    float*                  losses,
    CUDAContext*            ctx) {
    _SoftmaxCrossEntropy
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, prob, targets, losses
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA