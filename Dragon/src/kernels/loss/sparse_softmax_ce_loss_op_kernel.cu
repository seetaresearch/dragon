#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SparseSoftmaxCrossEntropy <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SparseSoftmaxCrossEntropy(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    Tx*                     losses,
    int*                    flags) {
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
            losses[idx] = -log(
                max(prob[(oix * axis_dim + label)
                    * inner_dim + iix], FLT_MIN)
            );
            flags[idx] = 1;
        }
    }
}

/*! SparseSoftmaxCrossEntropy <Tx = float32, Ty = float32, Device = CUDA> */

template <> void SparseSoftmaxCrossEntropy<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropy<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            prob, labels, ignores, losses, flags);
}

/*! SparseSoftmaxCrossEntropy <Tx = float32, Ty = int64, Device = CUDA> */

template <> void SparseSoftmaxCrossEntropy<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropy<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            prob, labels, ignores, losses, flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SparseSoftmaxCrossEntropyGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    Tx*                     dx,
    int*                    flags) {
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
            dx[(oix * axis_dim + label) * inner_dim + iix] -= 1;
            flags[idx] = 1;
        }
    }
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float32, Ty = float32, Device = CUDA> */

template<> void SparseSoftmaxCrossEntropyGrad<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyGrad<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            prob, labels, ignores, dx, flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float32, Ty = int64, Device = CUDA> */

template<> void SparseSoftmaxCrossEntropyGrad<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyGrad<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            prob, labels, ignores, dx, flags);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA