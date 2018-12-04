#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SparseSoftmaxCrossEntropy <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SparseSoftmaxCrossEntropy(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    Tx*                     losses,
    Tx*                     flags) {
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
                losses[idx] = -std::log(std::max(prob[t], FLT_MIN));
                flags[idx] = 1;
            }
        }
    }
}

/*! SparseSoftmaxCrossEntropy <Tx = float32, Ty = float32, Device = CPU> */

template <> void SparseSoftmaxCrossEntropy<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropy<float, float>(
        outer_dim, axis_dim, inner_dim,
            prob, labels, ignores, num_ignores,
                losses, flags);
}

/*! SparseSoftmaxCrossEntropy <Tx = float32, Ty = int64, Device = CPU> */

template <> void SparseSoftmaxCrossEntropy<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropy<float, int64_t>(
        outer_dim, axis_dim, inner_dim,
            prob, labels, ignores, num_ignores,
                losses, flags);
}

/*! SparseSoftmaxCrossEntropy <Tx = float16, Ty = float32, Device = CPU> */

template <> void SparseSoftmaxCrossEntropy<float16, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! SparseSoftmaxCrossEntropy <Tx = float16, Ty = int64, Device = CPU> */

template <> void SparseSoftmaxCrossEntropy<float16, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _SparseSoftmaxCrossEntropyGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    Tx*                     dx,
    Tx*                     flags) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const int idx = oix * inner_dim + iix;
            const int label = (int)labels[idx];
            int k;
            for (k = 0; k < num_ignores; ++k)
                if (label == ignores[k]) break;
            if (k != num_ignores) {
                for (int c = 0; c < axis_dim; ++c)
                    dx[(oix * axis_dim + c) * inner_dim + iix] = 0;
                flags[idx] = 0;
            } else {
                dx[(oix * axis_dim + label) * inner_dim + iix] -= 1;
                flags[idx] = 1;
            }
        }
    }
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float32, Ty = float32, Device = CPU> */

template<> void SparseSoftmaxCrossEntropyGrad<float, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropyGrad<float, float>(
        outer_dim, axis_dim, inner_dim,
            prob, labels, ignores,
                num_ignores, dx, flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float32, Ty = int64, Device = CPU> */

template<> void SparseSoftmaxCrossEntropyGrad<float, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropyGrad<float, int64_t>(
        outer_dim, axis_dim, inner_dim,
            prob, labels, ignores,
                num_ignores, dx, flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float16, Ty = float32, Device = CPU> */

template<> void SparseSoftmaxCrossEntropyGrad<float16, float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float16*                dx,
    float*                  flags,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float16, Ty = int64, Device = CPU> */

template<> void SparseSoftmaxCrossEntropyGrad<float16, int64_t, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float16*                dx,
    float*                  flags,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon