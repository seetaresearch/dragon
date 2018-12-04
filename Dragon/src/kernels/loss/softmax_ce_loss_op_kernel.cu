  #ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SoftmaxCrossEntropy <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SoftmaxCrossEntropy(
    const int               count,
    const T*                prob,
    const T*                target,
    T*                      loss) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        loss[idx] = -target[idx] * log(max(prob[idx], FLT_MIN));
    }
}

template <> void SoftmaxCrossEntropy<float, CUDAContext>(
    const int               count,
    const float*            prob,
    const float*            target,
    float*                  loss,
    CUDAContext*            ctx) {
    _SoftmaxCrossEntropy<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, prob, target, loss);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA