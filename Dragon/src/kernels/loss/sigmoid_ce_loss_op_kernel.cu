#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SigmoidCrossEntropy <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidCrossEntropy(
    const int               count,
    const T*                logits,
    const T*                targets,
    T*                      losses,
    int*                    flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (targets[idx] < 0) {
            losses[idx] = flags[idx] = 0;
        } else {
            losses[idx] = log(1 +
                exp(logits[idx] - 2 * logits[idx] * (logits[idx] >= 0))
            ) + logits[idx] * ((logits[idx] >= 0) - targets[idx]);
            flags[idx] = 1;
        }
    }
}

template <> void SigmoidCrossEntropy<float, CUDAContext>(
    const int               count,
    const float*            logits,
    const float*            targets,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    _SigmoidCrossEntropy<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, logits, targets, losses, flags);
}

/*! SigmoidCrossEntropyGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidCrossEntropyGrad(
    const int               count,
    const T*                logits,
    const T*                targets,
    T*                      dlogits,
    int*                    flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (targets[idx] < 0) {
            dlogits[idx] = flags[idx] = 0;
        } else {
            dlogits[idx] = 1 / (1 + exp(-logits[idx])) - targets[idx];
            flags[idx] = 1;
        }
    }
}

template <> void SigmoidCrossEntropyGrad<float, CUDAContext>(
    const int               count,
    const float*            logits,
    const float*            targets,
    float*                  dlogits,
    int*                    flags,
    CUDAContext*            ctx) {
    _SigmoidCrossEntropyGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, logits, targets, dlogits, flags);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA