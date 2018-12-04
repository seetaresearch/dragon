#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SigmoidFocalLoss <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidFocalLoss(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                logits,
    const T*                targets,
    T*                      losses,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int iix = idx % inner_dim;
        const int aix = (idx / inner_dim) % axis_dim;
        const int oix = idx / inner_dim / axis_dim;
        const int t = targets[oix * inner_dim + iix];
        //  ``0`` is reserved for targets if neg id is zero
        //  use ``aix + 1`` to match the targets
        T c1 = (t == (aix + (neg_id ? 0 : 1)));
        T c2 = (t != -1) & (t != (aix + (neg_id ? 0 : 1)));
        T p = 1 / (1 + exp(-logits[idx]));  //  logit -> prob

        // (1 - p)^{gamma} * log(p)
        T pos_term = pow(1 - p, gamma) * log(max(p, FLT_MIN));

        // p^{gamma} * log(1 - p)
        T neg_term = pow(p, gamma) * (
            -logits[idx] * (logits[idx] >= 0) - log(
                1 + exp(logits[idx] - 2 * logits[idx] * (logits[idx] >= 0)))
       );

        losses[idx] = (T)0;
        losses[idx] += -c1 * pos_term * pos_alpha;
        losses[idx] += -c2 * neg_term * neg_alpha;
        flags[idx] = c1;
    }
}

template <> void SigmoidFocalLoss<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const float*            targets,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const auto count = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLoss<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                logits, targets, losses, flags);
}

/*! SigmoidFocalLossGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SigmoidFocalLossGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                logits,
    const T*                targets,
    T*                      dlogits,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int iix = idx % inner_dim;
        const int aix = (idx / inner_dim) % axis_dim;
        const int oix = idx / inner_dim / axis_dim;
        const int t = targets[oix * inner_dim + iix];
        //  ``0`` is reserved for targets if neg id is zero
        //  use ``aix + 1`` to match the targets
        T c1 = (t == (aix + (neg_id ? 0 : 1)));
        T c2 = (t != -1) & (t != (aix + (neg_id ? 0 : 1)));
        T p = 1 / (1 + exp(-logits[idx]));  //  logit -> prob

        // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
        T pos_term = pow((1 - p), gamma) * (
            1 - p - p * gamma * log(max(p, FLT_MIN))
        );

        // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
        T neg_term = pow(p, gamma) * (
            (-logits[idx] * (logits[idx] >= 0) - log(
                1 + exp(logits[idx] - 2 * logits[idx] * (logits[idx] >= 0)))
            ) * (1 - p) * gamma - p
        );

        dlogits[idx] = (T)0;
        dlogits[idx] += -c1 * pos_term * pos_alpha;
        dlogits[idx] += -c2 * neg_term * neg_alpha;
        flags[idx] = c1;
    }
}

template <> void SigmoidFocalLossGrad<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const float*            targets,
    float*                  dlogits,
    float*                  flags,
    CUDAContext*            ctx) {
    const auto count = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLossGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                logits, targets, dlogits, flags);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA