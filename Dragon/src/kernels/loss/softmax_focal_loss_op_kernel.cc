#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SoftmaxFocalLoss <T = float32, Device = CPU> */

template <> void SoftmaxFocalLoss<float, CPUContext>(
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
    CPUContext*             ctx) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int idx = oix * inner_dim + iix;
            const int label = (int)labels[idx];
            int k;
            for (k = 0; k < num_ignores; ++k) {
                if (label == ignores[k]) {
                    losses[idx] = flags[idx] = 0.f;
                    break;
                }
            }
            if (k == num_ignores) {
                const int t = (oix * axis_dim + label) * inner_dim + iix;
                float labeled_prob = std::max(labeled_prob, FLT_MIN);
                float scale = std::pow((1.f - prob[t]), gamma);
                scale = label > neg_id ?
                    pos_alpha * scale :  neg_alpha * scale;
                losses[idx] = -scale * std::log(labeled_prob);
                flags[idx] = label > neg_id ? 1.f : 0.f;
            }
        }
    }
}

/*! SoftmaxFocalLossGrad <T = float32, Device = CPU> */

template<> void SoftmaxFocalLossGrad<float, CPUContext>(
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
    CPUContext*             ctx) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int idx = oix * inner_dim + iix;
            const int label = (int)labels[idx];
            int k;
            for (k = 0; k < num_ignores; ++k)
                if (label == ignores[k]) break;
            if (k != num_ignores) {
                for (int c = 0; c < axis_dim; ++c)
                    dx[(oix * axis_dim + c) * inner_dim + iix] = 0.f;
                flags[idx] = 0.f;
            } else {
                const int t = (oix * axis_dim + label) * inner_dim + iix;
                float onemp = 1.f - prob[t];
                // Unstable if gamma is 0
                float grad = -gamma * pow(onemp, gamma - 1)
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
                flags[idx] = label > neg_id ? 1.f : 0.f;
            }
        }
    }
}

}  // namespace kernel

}  // namepsace dragon