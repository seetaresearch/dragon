#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SigmoidFocalLoss <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SigmoidFocalLoss(
    const int               outer_dim,
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
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int aix = 0; aix < axis_dim; ++aix) {
            int64_t offset = oix * axis_dim + aix;
            for (int iix = 0; iix < inner_dim; ++iix) {
                const int64_t i = offset * inner_dim + iix;
                const int t = (int)targets[oix * inner_dim + iix];
                // ``0`` is reserved for targets if neg id is zero
                // Use ``aix + 1`` to match the targets
                Tx c1 = (float)(t == (aix + (neg_id ? 0 : 1)));
                Tx c2 = (float)((t != -1) & (t != (aix + (neg_id ? 0 : 1))));
                Tx p = 1 / (1 + std::exp(-logits[i]));  // logit -> prob

                // (1 - p)^{gamma} * log(p)
                Tx pos_term = std::pow(1 - p, gamma) * (
                    std::log(std::max(p, FLT_MIN))
                );

                // p^{gamma} * log(1 - p)
                Tx neg_term = std::pow(p, gamma) * (
                    -logits[i] * (logits[i] >= 0) - std::log(
                        1 + std::exp(logits[i] - 2 * logits[i] * (logits[i] >= 0)))
                );

                losses[i] = (Tx)0;
                losses[i] += -c1 * pos_term * pos_alpha;
                losses[i] += -c2 * neg_term * neg_alpha;
                flags[i] = c1;
            }
        }
    }
}

/*! SigmoidFocalLoss <Tx = float32, Ty = float32, Device = CPU> */

template <> void SigmoidFocalLoss<float, float, CPUContext>(
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
    CPUContext*             ctx) {
    _SigmoidFocalLoss<float, float>(
        outer_dim, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                logits, targets, losses, flags);
}

/*! SigmoidFocalLoss <Tx = float32, Ty = int64, Device = CPU> */

template <> void SigmoidFocalLoss<float, int64_t, CPUContext>(
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
    CPUContext*             ctx) {
    _SigmoidFocalLoss<float, int64_t>(
        outer_dim, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                logits, targets, losses, flags);
}

/*! SigmoidFocalLossGrad <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SigmoidFocalLossGrad(
    const int               outer_dim,
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
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int aix = 0; aix < axis_dim; ++aix) {
            int64_t offset = oix * axis_dim + aix;
            for (int iix = 0; iix < inner_dim; ++iix) {
                const int64_t i = offset * inner_dim + iix;
                const int t = (int)targets[oix * inner_dim + iix];
                // ``0`` is reserved for targets if neg id is zero
                // Use ``aix + 1`` to match the targets
                Tx c1 = (float)(t == (aix + (neg_id ? 0 : 1)));
                Tx c2 = (float)((t != -1) & (t != (aix + (neg_id ? 0 : 1))));
                Tx p = 1 / (1 + std::exp(-logits[i]));  // logit -> prob

                // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
                Tx pos_term = std::pow((1 - p), gamma) * (
                    1 - p - p * gamma * std::log(std::max(p, FLT_MIN))
                );

                // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
                Tx neg_term = std::pow(p, gamma) * (
                    (-logits[i] * (logits[i] >= 0) - log(
                        1 + exp(logits[i] - 2 * logits[i] * (logits[i] >= 0)))
                    ) * (1 - p) * gamma - p
                );

                dlogits[i] = (Tx)0;
                dlogits[i] += -c1 * pos_term * pos_alpha;
                dlogits[i] += -c2 * neg_term * neg_alpha;
                flags[i] = c1;
            }
        }
    }
}

/*! SigmoidFocalLossGrad <Tx = float32, Ty = float32, Device = CPU> */

template <> void SigmoidFocalLossGrad<float, float, CPUContext>(
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
    CPUContext*             ctx) {
    _SigmoidFocalLossGrad<float, float>(
        outer_dim, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                logits, targets, dlogits, flags);
}

/*! SigmoidFocalLossGrad <Tx = float32, Ty = int64_t, Device = CPU> */

template <> void SigmoidFocalLossGrad<float, int64_t, CPUContext>(
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
    CPUContext*             ctx) {
    _SigmoidFocalLossGrad<float, int64_t>(
        outer_dim, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                logits, targets, dlogits, flags);
}

}  // namespace kernel

}  // namepsace dragon