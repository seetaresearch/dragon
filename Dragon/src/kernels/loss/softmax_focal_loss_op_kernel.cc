#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SoftmaxFocalLoss(
    const int               outer_dim,
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
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int i = oix * inner_dim + iix;
            const int label = (int)labels[i];
            int k;
            for (k = 0; k < nignores; ++k) {
                if (label == ignores[k]) {
                    losses[i] = flags[i] = 0;
                    break;
                }
            }
            if (k == nignores) {
                const int t = (
                    oix * axis_dim + label
                        ) * inner_dim + iix;
                Tx labeled_prob = std::max(labeled_prob, FLT_MIN);
                Tx scale = std::pow(Tx(1) - prob[t], gamma);
                scale = label > neg_id ?
                    pos_alpha * scale :  neg_alpha * scale;
                losses[i] = -scale * std::log(labeled_prob);
                flags[i] = label > neg_id ? 1 : 0;
            }
        }
    }
}

/* <Tx = float32, Ty = float32, Device = CPU> */

template <> void SoftmaxFocalLoss<float, float, CPUContext>(
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
    CPUContext*             ctx) {
    _SoftmaxFocalLoss(
        outer_dim, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, losses, flags
    );
}

/* <Tx = float32, Ty = int64, Device = CPU> */

template <> void SoftmaxFocalLoss<float, int64_t, CPUContext>(
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
    CPUContext*             ctx) {
    _SoftmaxFocalLoss(
        outer_dim, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, losses, flags
    );
}

/* <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SoftmaxFocalLossGrad(
    const int               outer_dim,
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
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int i = oix * inner_dim + iix;
            const int label = (int)labels[i];
            int k;
            for (k = 0; k < nignores; ++k)
                if (label == ignores[k]) break;
            if (k != nignores) {
                for (int c = 0; c < axis_dim; ++c)
                    dx[(oix * axis_dim + c
                          ) * inner_dim + iix
                    ] = Tx(0);
                flags[i] = 0;
            } else {
                const int t = (
                    oix * axis_dim + label
                        ) * inner_dim + iix;
                Tx onemp = Tx(1) - prob[t];
                // Unstable if gamma is 0
                Tx grad = -gamma * pow(onemp, gamma - Tx(1))
                                 * log(std::max(prob[t], FLT_MIN))
                                 * prob[t] + pow(onemp, gamma);
                grad = label > neg_id ?
                    pos_alpha * grad : neg_alpha * grad;
                for (int c = 0; c < axis_dim; ++c) {
                    const int xi = (
                        oix * axis_dim + c
                            ) * inner_dim + iix;
                    if (c == label) {
                        dx[xi] = grad * (prob[t] - Tx(1));
                    } else {
                        dx[xi] = grad * prob[xi];
                    }
                }
                flags[i] = label > neg_id ? 1 : 0;
            }
        }
    }
}

/* <Tx = float32, Ty = float32, Device = CPU> */

template<> void SoftmaxFocalLossGrad<float, float, CPUContext>(
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
    CPUContext*             ctx) {
    _SoftmaxFocalLossGrad(
        outer_dim, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, dx, flags
    );
}

/* <Tx = float32, Ty = int64, Device = CPU> */

template<> void SoftmaxFocalLossGrad<float, int64_t, CPUContext>(
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
    CPUContext*             ctx) {
    _SoftmaxFocalLossGrad(
        outer_dim, axis_dim, inner_dim,
        pos_alpha, neg_alpha, gamma, neg_id,
        nignores, ignores,
        prob, labels, dx, flags
    );
}

}  // namespace kernel

}  // namepsace dragon