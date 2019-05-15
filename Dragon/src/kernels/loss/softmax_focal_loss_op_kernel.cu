#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SoftmaxFocalLoss <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SoftmaxFocalLoss(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const Tx*               prob,
    const Ty*               labels,
    Tx*                     losses,
    int*                    flags) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int oix = i / inner_dim;
        const int iix = i % inner_dim;
        const int label = labels[oix * inner_dim + iix];
        int k;
        for (k = 0; k < nignores; k++) {
            if (label == ignores[k]) {
                losses[i] = flags[i] = 0;
                break;
            }
        }
        if (k == nignores) {
            const int t = (
                oix * axis_dim + label
                    ) * inner_dim + iix;
            Tx scale = pow(Tx(1) - prob[t], gamma);
            scale = label > neg_id ?
                pos_alpha * scale : neg_alpha * scale;
            losses[i] = -scale * log(max(prob[t], FLT_MIN));
            flags[i] = label > neg_id ? 1 : 0;
        }
    }
}

/*! SoftmaxFocalLoss <Tx = float32, Ty = float32, Device = CUDA> */

template <> void SoftmaxFocalLoss<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const float*            prob,
    const float*            labels,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    auto num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLoss
        <<< CUDA_BLOCKS(num_preds), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        num_preds, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, losses, flags
    );
}

/*! SoftmaxFocalLoss <Tx = float32, Ty = int64, Device = CUDA> */

template <> void SoftmaxFocalLoss<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const float*            prob,
    const int64_t*          labels,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    auto num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLoss
        <<< CUDA_BLOCKS(num_preds), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        num_preds, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, losses, flags
    );
}

/*! SoftmaxFocalLossGrad <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SoftmaxFocalLossGrad(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const Tx*               prob,
    const Ty*               labels,
    Tx*                     dx,
    int*                    flags) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int oix = i / inner_dim;
        const int iix = i % inner_dim;
        const int label = labels[oix * inner_dim + iix];
        int k;
        for (k = 0; k < nignores; k++)
            if (label == ignores[k]) break;
        if (k != nignores) {
            for (int c = 0; c < axis_dim; c++)
                dx[(oix * axis_dim + c
                      ) * inner_dim + iix
                ] = (Tx)0;
            flags[i] = 0;
        } else {
            const int t = (
                oix * axis_dim + label
                    ) * inner_dim + iix;
            Tx onemp = Tx(1) - prob[t];
            // Unstable if gamma is 0
            Tx grad = -gamma * pow(onemp, gamma - Tx(1))
                             * log(max(prob[t], FLT_MIN))
                             * prob[t] + pow(onemp, gamma);
            grad = label > neg_id ?
                pos_alpha * grad : neg_alpha * grad;
            for (int c = 0; c < axis_dim; c++) {
                const int xi = (
                    oix * axis_dim + c
                        ) * inner_dim + iix;
                if (c == label) {
                    dx[xi] = grad * (prob[t] - 1);
                } else {
                    dx[xi] = grad * prob[xi];
                }
            }
            flags[i] = label > neg_id ? 1 : 0;
        }
    }
}

/*! SoftmaxFocalLossGrad <Tx = float32, Ty = float32, Device = CUDA> */

template<> void SoftmaxFocalLossGrad<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const float*            prob,
    const float*            labels,
    float*                  dx,
    int*                    flags,
    CUDAContext*            ctx) {
    auto num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLossGrad
        <<< CUDA_BLOCKS(num_preds), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        num_preds, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, dx, flags
    );
}

/*! SoftmaxFocalLossGrad <Tx = float32, Ty = int64, Device = CUDA> */

template<> void SoftmaxFocalLossGrad<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const float*            prob,
    const int64_t*          labels,
    float*                  dx,
    int*                    flags,
    CUDAContext*            ctx) {
    auto num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLossGrad
        <<< CUDA_BLOCKS(num_preds), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        num_preds, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, dx, flags
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA