#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! NLLLoss <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _NLLLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const Tx*               log_prob,
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
                losses[idx] = -log_prob[
                    (oix * axis_dim + label) * inner_dim + iix];
                flags[idx] = 1;
            }
        }
    }
}

/*! NLLLoss <Tx = float32, Ty = float32, Device = CPU> */

template <> void NLLLoss<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const float*            labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CPUContext*             ctx) {
    _NLLLoss<float, float>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, losses, flags);
}

/*! NLLLoss <Tx = float32, Ty = int64, Device = CPU> */

template <> void NLLLoss<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CPUContext*             ctx) {
    _NLLLoss<float, int64_t>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, losses, flags);
}

/*! NLLLossGrad <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _NLLLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const Tx*               log_prob,
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
                flags[idx] = 0;
            } else {
                dx[(oix * axis_dim + label) * inner_dim + iix] = -1;
                flags[idx] = 1;
            }
        }
    }
}

/*! NLLLossGrad <Tx = float32, Ty = float32, Device = CPU> */

template<> void NLLLossGrad<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const float*            labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CPUContext*             ctx) {
    _NLLLossGrad<float, float>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, dx, flags);
}

/*! NLLLossGrad <Tx = float32, Ty = int64, Device = CPU> */

template<> void NLLLossGrad<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CPUContext*             ctx) {
    _NLLLossGrad<float, int64_t>(
        outer_dim, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, dx, flags);
}

}  // namespace kernel

}  // namepsace dragon