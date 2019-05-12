#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _NLLLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               log_prob,
    const Ty*               target,
    Tx*                     loss,
    int*                    flag) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int i = oix * inner_dim + iix;
            const int label = (int)target[i];
            int k;
            for (k = 0; k < nignores; ++k) {
                if (label == ignore[k]) {
                    loss[i] = flag[i] = 0;
                    break;
                }
            }
            if (k == nignores) {
                loss[i] = -log_prob[
                    (oix * axis_dim + label
                       ) * inner_dim + iix];
                flag[i] = 1;
            }
        }
    }
}

/*! <Tx = float32, Ty = float32, Device = CPU> */

template <> void NLLLoss<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const float*            target,
    float*                  loss,
    int*                    flag,
    CPUContext*             ctx) {
    _NLLLoss(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, loss, flag
    );
}

/*! <Tx = float32, Ty = int64, Device = CPU> */

template <> void NLLLoss<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const int64_t*          target,
    float*                  loss,
    int*                    flag,
    CPUContext*             ctx) {
    _NLLLoss(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, loss, flag
    );
}

/*! <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _NLLLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               log_prob,
    const Ty*               target,
    Tx*                     dx,
    int*                    flag) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int i = oix * inner_dim + iix;
            const int label = (int)target[i];
            int k;
            for (k = 0; k < nignores; ++k)
                if (label == ignore[k]) break;
            if (k != nignores) {
                flag[i] = 0;
            } else {
                dx[(oix * axis_dim + label
                      ) * inner_dim + iix] = -1;
                flag[i] = 1;
            }
        }
    }
}

/*! <Tx = float32, Ty = float32, Device = CPU> */

template<> void NLLLossGrad<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const float*            target,
    float*                  dx,
    int*                    flag,
    CPUContext*             ctx) {
    _NLLLossGrad(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, dx, flag
    );
}

/*! <Tx = float32, Ty = int64, Device = CPU> */

template<> void NLLLossGrad<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const int64_t*          target,
    float*                  dx,
    int*                    flag,
    CPUContext*             ctx) {
    _NLLLossGrad(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, dx, flag
    );
}

}  // namespace kernel

}  // namepsace dragon