#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidCrossEntropy(
    const int               nthreads,
    const T*                logit,
    const T*                target,
    T*                      loss,
    int*                    flag) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        if (target[i] < 0) {
            loss[i] = flag[i] = 0;
        } else {
            loss[i] = log(
                T(1) + exp(
                    logit[i] - T(2) * logit[i] * (
                            logit[i] >= 0
                        )
                )
            ) + logit[i] * (
                (logit[i] >= 0) - target[i]
            );
            flag[i] = 1;
        }
    }
}

template <> void SigmoidCrossEntropy<float, CUDAContext>(
    const int               count,
    const float*            logit,
    const float*            target,
    float*                  loss,
    int*                    flag,
    CUDAContext*            ctx) {
    _SigmoidCrossEntropy
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, logit, target, loss, flag
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidCrossEntropyGrad(
    const int               nthreads,
    const T*                logit,
    const T*                target,
    T*                      dlogit,
    int*                    flag) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        if (target[i] < 0) {
            dlogit[i] = flag[i] = 0;
        } else {
            dlogit[i] = T(1) / (
                T(1) + exp(-logit[i])
            ) - target[i];
            flag[i] = 1;
        }
    }
}

template <> void SigmoidCrossEntropyGrad<float, CUDAContext>(
    const int               count,
    const float*            logit,
    const float*            target,
    float*                  dlogit,
    int*                    flag,
    CUDAContext*            ctx) {
    _SigmoidCrossEntropyGrad
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, logit, target, dlogit, flag
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA