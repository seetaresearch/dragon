#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SoftmaxFocalLoss <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SoftmaxFocalLoss(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                prob,
    const T*                labels,
    const int*              ignores,
    const int               num_ignores,
    T*                      losses,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        const int label = labels[oix * inner_dim + iix];
        int k;
        for (k = 0; k < num_ignores; k++) {
            if (label == ignores[k]) {
                losses[idx] = flags[idx] = 0;
                break;
            }
        }
        if (k == num_ignores) {
            const int t = (oix * axis_dim + label) * inner_dim + iix;
            T scale = pow(1.f - prob[t], gamma);
            scale = label > neg_id ?
                pos_alpha * scale : neg_alpha * scale;
            losses[idx] = -scale * log(max(prob[t], FLT_MIN));
            flags[idx] = label > neg_id ? 1 : 0;
        }
    }
}

template <> void SoftmaxFocalLoss<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLoss<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                prob, labels, ignores, num_ignores,
                    losses, flags);
}

/*! SoftmaxFocalLossGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SoftmaxFocalLossGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                prob,
    const T*                labels,
    const int*              ignores,
    const int               num_ignores,
    T*                      dx,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        const int label = labels[oix * inner_dim + iix];
        int k;
        for (k = 0; k < num_ignores; k++)
            if (label == ignores[k]) break;
        if (k != num_ignores) {
            for (int c = 0; c < axis_dim; c++)
                dx[(oix * axis_dim + c) * inner_dim + iix] = 0;
            flags[idx] = 0;
        } else {
            const int t = (oix * axis_dim + label) * inner_dim + iix;
            T onemp = 1. - prob[t];
            //  unstable if gamma is 0
            T grad = -gamma * pow(onemp, gamma - 1)
                            * log(max(prob[t], FLT_MIN))
                            * prob[t] + pow(onemp, gamma);
            grad = label > neg_id ?
                pos_alpha * grad : neg_alpha * grad;
            for (int c = 0; c < axis_dim; c++) {
                const int i = (oix * axis_dim + c) * inner_dim + iix;
                if (c == label) {
                    dx[i] = grad * (prob[t] - 1);
                } else {
                    dx[i] = grad * prob[i];
                }
            }
            flags[idx] = label > neg_id ? 1 : 0;
        }
    }
}

template<> void SoftmaxFocalLossGrad<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLossGrad<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                prob, labels, ignores, num_ignores,
                    dx, flags);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA