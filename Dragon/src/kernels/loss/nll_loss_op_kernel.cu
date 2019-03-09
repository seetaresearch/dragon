#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! NLLLoss <Tx = float32, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _NLLLoss(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const Tx*               log_prob,
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
            losses[idx] = -log_prob[
                (oix * axis_dim + label) * inner_dim + iix];
            flags[idx] = 1;
        }
    }
}

/*! NLLLoss <Tx = float32, Ty = float32, Device = CUDA> */

template <> void NLLLoss<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const float*            labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _NLLLoss<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, losses, flags);
}

/*! NLLLoss <Tx = float32, Ty = int64, Device = CUDA> */

template <> void NLLLoss<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  losses,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _NLLLoss<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, losses, flags);
}

/*! NLLLossGrad <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _NLLLossGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const Tx*               log_prob,
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
            flags[idx] = 0;
        } else {
            dx[(oix * axis_dim + label) * inner_dim + iix] = -1;
            flags[idx] = 1;
        }
    }
}

/*! NLLLossGrad <Tx = float32, Ty = float32, Device = CUDA> */

template<> void NLLLossGrad<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const float*            labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _NLLLossGrad<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, dx, flags);
}

/*! NLLLossGrad <Tx = float32, Ty = int64, Device = CUDA> */

template<> void NLLLossGrad<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               num_ignores,
    const float*            log_prob,
    const int64_t*          labels,
    const int*              ignores,
    float*                  dx,
    int*                    flags,
    CUDAContext*            ctx) {
    const auto num_preds = outer_dim * inner_dim;
    _NLLLossGrad<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim, num_ignores,
            log_prob, labels, ignores, dx, flags);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA