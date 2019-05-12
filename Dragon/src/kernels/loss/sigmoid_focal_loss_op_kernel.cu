#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SigmoidFocalLoss <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SigmoidFocalLoss(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const Tx*               logits,
    const Ty*               targets,
    Tx*                     losses,
    int*                    flags) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int iix = i % inner_dim;
        const int aix = (i / inner_dim) % axis_dim;
        const int oix = i / inner_dim / axis_dim;
        const int t = targets[oix * inner_dim + iix];
        // *0* is reserved for targets if neg id is zero
        // Use *aix + 1* to match the targets
        Tx c1 = (Tx)(t == (aix + (neg_id ? 0 : 1)));
        Tx c2 = (Tx)((t != -1) & (t != (aix + (neg_id ? 0 : 1))));
        Tx p = Tx(1) / (Tx(1) + exp(-logits[i]));

        // (1 - p)^{gamma} * log(p)
        Tx pos_term = pow(Tx(1) - p, gamma) * log(max(p, FLT_MIN));

        // p^{gamma} * log(1 - p)
        Tx neg_term = pow(p, gamma) * (
            -logits[i] * (logits[i] >= 0) - log(
                Tx(1) + exp(
                    logits[i] - 2 * logits[i] * (
                            logits[i] >= 0
                        )
                )
            )
       );

        losses[i] = Tx(0);
        losses[i] += -c1 * pos_term * pos_alpha;
        losses[i] += -c2 * neg_term * neg_alpha;
        flags[i] = c1;
    }
}

/*! SigmoidFocalLoss <Tx = float32, Ty = float32, Device = CUDA> */

template <> void SigmoidFocalLoss<float, float, CUDAContext>(
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
    int*                    flags,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLoss
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nthreads, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        logits, targets, losses, flags
    );
}

/*! SigmoidFocalLoss <Tx = float32, Ty = int64, Device = CUDA> */

template <> void SigmoidFocalLoss<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const int64_t*          targets,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLoss
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nthreads, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        logits, targets, losses, flags
    );
}

/*! SigmoidFocalLossGrad <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SigmoidFocalLossGrad(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const Tx*               logits,
    const Ty*               targets,
    Tx*                     dlogits,
    int*                    flags) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int iix = i % inner_dim;
        const int aix = (i / inner_dim) % axis_dim;
        const int oix = i / inner_dim / axis_dim;
        const int t = targets[oix * inner_dim + iix];
        // *0* is reserved for targets if neg id is zero
        // Use *aix + 1* to match the targets
        Tx c1 = (Tx)(t == (aix + (neg_id ? 0 : 1)));
        Tx c2 = (Tx)((t != -1) & (t != (aix + (neg_id ? 0 : 1))));
        Tx p = Tx(1) / (Tx(1) + exp(-logits[i]));

        // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
        Tx pos_term = pow(Tx(1) - p, gamma) * (
            Tx(1) - p - p * gamma * log(max(p, FLT_MIN))
        );

        // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
        Tx neg_term = pow(p, gamma) * (
            (-logits[i] * (logits[i] >= 0) - log(
                Tx(1) + exp(
                    logits[i] - Tx(2) * logits[i] * (
                            logits[i] >= 0
                        )
                    )
                )
            ) * (1 - p) * gamma - p
        );

        dlogits[i] = Tx(0);
        dlogits[i] += -c1 * pos_term * pos_alpha;
        dlogits[i] += -c2 * neg_term * neg_alpha;
        flags[i] = c1;
    }
}

/*! SigmoidFocalLossGrad <Tx = float32, Ty = float32, Device = CUDA> */

template <> void SigmoidFocalLossGrad<float, float, CUDAContext>(
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
    int*                    flags,
    CUDAContext*            ctx) {
    auto count = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLossGrad
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        logits, targets, dlogits, flags
    );
}

/*! SigmoidFocalLossGrad <Tx = float32, Ty = int64, Device = CUDA> */

template <> void SigmoidFocalLossGrad<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const int64_t*          targets,
    float*                  dlogits,
    int*                    flags,
    CUDAContext*            ctx) {
    auto count = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLossGrad
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        logits, targets, dlogits, flags
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA