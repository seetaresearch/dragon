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
    const Tx*               log_prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    Tx*                     losses,
    Tx*                     flags) {
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
    const float*            log_prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLoss<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            log_prob, labels, ignores,
                num_ignores, losses, flags);
}

/*! NLLLoss <Tx = float32, Ty = int64, Device = CUDA> */

template <> void NLLLoss<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            log_prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLoss<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            log_prob, labels, ignores,
                num_ignores, losses, flags);
}

/*! NLLLoss <Tx = float16, Ty = ?, Device = CUDA> */

template <typename Ty>
__global__ void _NLLLossHalf(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const half*             log_prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        const int label = labels[oix * inner_dim + iix];
        int k;
        for (k = 0; k < num_ignores; k++) {
            if (label == ignores[k]) {
                losses[idx] = flags[idx] = 0.f;
                break;
            }
        }
        if (k == num_ignores) {
            losses[idx] = __half2float(__hneg(
                log_prob[(oix * axis_dim + label) * inner_dim + iix]));
            flags[idx] = 1.f;
        }
#endif
    }
}

/*! NLLLoss <Tx = float16, Ty = float32, Device = CUDA> */

template <> void NLLLoss<float16, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          log_prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLossHalf<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(log_prob), labels,
                ignores, num_ignores, losses, flags);
}

/*! NLLLoss <Tx = float16, Ty = int64, Device = CUDA> */

template <> void NLLLoss<float16, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          log_prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLossHalf<int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(log_prob), labels,
                ignores, num_ignores, losses, flags);
}

/*! NLLLossGrad <Tx = float32, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _NLLLossGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               log_prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    Tx*                     dx,
    Tx*                     flags) {
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
    const float*            log_prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLossGrad<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            log_prob, labels, ignores,
                num_ignores, dx, flags);
}

/*! NLLLossGrad <Tx = float32, Ty = int64, Device = CUDA> */

template<> void NLLLossGrad<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            log_prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLossGrad<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            log_prob, labels, ignores,
                num_ignores, dx, flags);
}

/*! NLLLossGrad <Tx = float16, Ty = ?, Device = CUDA> */

template <typename Ty>
__global__ void _NLLLossGradHalf(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const half*             log_prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    half*                   dx,
    float*                  flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        const int label = labels[oix * inner_dim + iix];
        int k;
        for (k = 0; k < num_ignores; k++)
            if (label == ignores[k]) break;
        if (k != num_ignores) {
            flags[idx] = 0.f;
        } else {
            dx[(oix * axis_dim + label) * inner_dim + iix] = __float2half(-1.);
            flags[idx] = 1.f;
        }
#endif
    }
}

/*! NLLLossGrad <Tx = float16, Ty = float32, Device = CUDA> */

template<> void NLLLossGrad<float16, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          log_prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float16*                dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLossGradHalf<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(log_prob), labels,
                ignores, num_ignores,
                    reinterpret_cast<half*>(dx), flags);
}

/*! NLLLossGrad <Tx = float16, Ty = int64, Device = CUDA> */

template<> void NLLLossGrad<float16, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          log_prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float16*                dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _NLLLossGradHalf<int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(log_prob), labels,
                ignores, num_ignores,
                    reinterpret_cast<half*>(dx), flags);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA