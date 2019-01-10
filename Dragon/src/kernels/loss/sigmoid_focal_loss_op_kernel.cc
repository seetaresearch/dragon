#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SigmoidFocalLoss <T = float32, Device = CPU> */

template <> void SigmoidFocalLoss<float, CPUContext>(
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
    CPUContext*             ctx) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int aix = 0; aix < axis_dim; ++aix) {
            int64_t offset = oix * axis_dim + aix;
            for (int iix = 0; iix < inner_dim; ++iix) {
                const int64_t i = offset * inner_dim + iix;
                const int t = (int)targets[oix * inner_dim + iix];
                // ``0`` is reserved for targets if neg id is zero
                // Use ``aix + 1`` to match the targets
                float c1 = (float)(t == (aix + (neg_id ? 0 : 1)));
                float c2 = (float)((t != -1) & (t != (aix + (neg_id ? 0 : 1))));
                float p = 1 / (1 + std::exp(-logits[i]));  // logit -> prob

                // (1 - p)^{gamma} * log(p)
                float pos_term = std::pow(1 - p, gamma) * (
                    std::log(std::max(p, FLT_MIN))
                );
    
                // p^{gamma} * log(1 - p)
                float neg_term = std::pow(p, gamma) * (
                    -logits[i] * (logits[i] >= 0) - std::log(
                        1 + std::exp(logits[i] - 2 * logits[i] * (logits[i] >= 0)))
                );

                losses[i] = 0.f;
                losses[i] += -c1 * pos_term * pos_alpha;
                losses[i] += -c2 * neg_term * neg_alpha;
                flags[i] = c1;
            }
        }
    }
}

/*! SigmoidFocalLossGrad <T = float32, Device = CPU> */

template <> void SigmoidFocalLossGrad<float, CPUContext>(
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
    CPUContext*             ctx) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int aix = 0; aix < axis_dim; ++aix) {
            int64_t offset = oix * axis_dim + aix;
            for (int iix = 0; iix < inner_dim; ++iix) {
                const int64_t i = offset * inner_dim + iix;
                const int t = (int)targets[oix * inner_dim + iix];
                // ``0`` is reserved for targets if neg id is zero
                // Use ``aix + 1`` to match the targets
                float c1 = (float)(t == (aix + (neg_id ? 0 : 1)));
                float c2 = (float)((t != -1) & (t != (aix + (neg_id ? 0 : 1))));
                float p = 1 / (1 + std::exp(-logits[i]));  // logit -> prob

                // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
                float pos_term = std::pow((1 - p), gamma) * (
                    1 - p - p * gamma * std::log(std::max(p, FLT_MIN))
                );

                // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
                float neg_term = std::pow(p, gamma) * (
                    (-logits[i] * (logits[i] >= 0) - log(
                        1 + exp(logits[i] - 2 * logits[i] * (logits[i] >= 0)))
                    ) * (1 - p) * gamma - p
                );

                dlogits[i] = 0.f;
                dlogits[i] += -c1 * pos_term * pos_alpha;
                dlogits[i] += -c2 * neg_term * neg_alpha;
                flags[i] = c1;
            }
        }
    }
}

}  // namespace kernel

}  // namepsace dragon