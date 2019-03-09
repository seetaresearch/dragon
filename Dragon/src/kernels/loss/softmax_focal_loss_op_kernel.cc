#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SoftmaxFocalLoss <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SoftmaxFocalLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    Tx*                     losses,
    int*                    flags) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int idx = oix * inner_dim + iix;
            const int label = (int)labels[idx];
            int k;
            for (k = 0; k < num_ignores; ++k) {
                if (label == ignores[k]) {
                    losses[idx] = flags[idx] = 0;
                    break;
                }
            }
            if (k == num_ignores) {
                const int t = (oix * axis_dim + label) * inner_dim + iix;
                Tx labeled_prob = std::max(labeled_prob, FLT_MIN);
                Tx scale = std::pow((1 - prob[t]), gamma);
                scale = label > neg_id ?
                    pos_alpha * scale :  neg_alpha * scale;
                losses[idx] = -scale * std::log(labeled_prob);
                flags[idx] = label > neg_id ? 1 : 0;
            }
        }
    }
}

/*! SoftmaxFocalLoss <Tx = float32, Ty = float32, Device = CPU> */

template <> void SoftmaxFocalLoss<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CPUContext*             ctx) {
    _SoftmaxFocalLoss<float, float>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            pos_alpha, neg_alpha, gamma, neg_id,
                prob, labels, ignores, losses, flags);
}

/*! SoftmaxFocalLoss <Tx = float32, Ty = int64, Device = CPU> */

template <> void SoftmaxFocalLoss<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CPUContext*             ctx) {
    _SoftmaxFocalLoss<float, int64_t>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            pos_alpha, neg_alpha, gamma, neg_id,
                prob, labels, ignores, losses, flags);
}

/*! SoftmaxFocalLossGrad <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SoftmaxFocalLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    Tx*                     dx,
    int*                    flags) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int idx = oix * inner_dim + iix;
            const int label = (int)labels[idx];
            int k;
            for (k = 0; k < num_ignores; ++k)
                if (label == ignores[k]) break;
            if (k != num_ignores) {
                for (int c = 0; c < axis_dim; ++c)
                    dx[(oix * axis_dim + c) * inner_dim + iix] = (Tx)0;
                flags[idx] = 0;
            } else {
                const int t = (oix * axis_dim + label) * inner_dim + iix;
                Tx onemp = 1 - prob[t];
                // Unstable if gamma is 0
                Tx grad = -gamma * pow(onemp, gamma - 1)
                                 * log(std::max(prob[t], FLT_MIN))
                                 * prob[t] + pow(onemp, gamma);
                grad = label > neg_id ?
                    pos_alpha * grad : neg_alpha * grad;
                for (int c = 0; c < axis_dim; ++c) {
                    const int i_ = (oix * axis_dim + c) * inner_dim + iix;
                    if (c == label) {
                        dx[i_] = grad * (prob[t] - 1);
                    } else {
                        dx[i_] = grad * prob[i_];
                    }
                }
                flags[idx] = label > neg_id ? 1 : 0;
            }
        }
    }
}

/*! SoftmaxFocalLossGrad <Tx = float32, Ty = float32, Device = CPU> */

template<> void SoftmaxFocalLossGrad<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CPUContext*             ctx) {
    _SoftmaxFocalLossGrad<float, float>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            pos_alpha, neg_alpha, gamma, neg_id,
                prob, labels, ignores, dx, flags);
}

/*! SoftmaxFocalLossGrad <Tx = float32, Ty = int64, Device = CPU> */

template<> void SoftmaxFocalLossGrad<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CPUContext*             ctx) {
    _SoftmaxFocalLossGrad<float, int64_t>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            pos_alpha, neg_alpha, gamma, neg_id,
                prob, labels, ignores, dx, flags);
}

}  // namespace kernel

}  // namepsace dragon