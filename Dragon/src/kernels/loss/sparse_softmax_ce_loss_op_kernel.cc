#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SparseSoftmaxCrossEntropy(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               prob,
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
                const int t = (
                    oix * axis_dim + label
                        ) * inner_dim + iix;
                loss[i] = -std::log(
                    std::max(prob[t], FLT_MIN));
                flag[i] = 1;
            }
        }
    }
}

/* <Tx = float32, Ty = float32, Device = CPU> */

template <> void SparseSoftmaxCrossEntropy<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            prob,
    const float*            target,
    float*                  loss,
    int*                    flag,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropy(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, prob, target, loss, flag
    );
}

/* <Tx = float32, Ty = int64, Device = CPU> */

template <> void SparseSoftmaxCrossEntropy<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            prob,
    const int64_t*          target,
    float*                  loss,
    int*                    flag,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropy(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, prob, target, loss, flag
    );
}

/* <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SparseSoftmaxCrossEntropyGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               prob,
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
                for (int c = 0; c < axis_dim; ++c)
                    dx[(oix * axis_dim + c
                          ) * inner_dim + iix] = Tx(0);
                flag[i] = 0;
            } else {
                dx[(oix * axis_dim + label
                      ) * inner_dim + iix] -= 1;
                flag[i] = 1;
            }
        }
    }
}

/* <Tx = float32, Ty = float32, Device = CPU> */

template<> void SparseSoftmaxCrossEntropyGrad<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            prob,
    const float*            target,
    float*                  dx,
    int*                    flag,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropyGrad(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, prob, target, dx, flag
    );
}

/* <Tx = float32, Ty = int64, Device = CPU> */

template<> void SparseSoftmaxCrossEntropyGrad<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            prob,
    const int64_t*          target,
    float*                  dx,
    int*                    flag,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropyGrad(
        outer_dim, axis_dim, inner_dim, nignores,
        ignore, prob, target, dx, flag
    );
}

}  // namespace kernel

}  // namepsace dragon