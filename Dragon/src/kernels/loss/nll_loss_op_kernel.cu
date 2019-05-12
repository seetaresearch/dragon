#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! <Tx = float32, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _NLLLoss(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               log_prob,
    const Ty*               target,
    Tx*                     loss,
    int*                    flag) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int oix = i / inner_dim;
        const int iix = i % inner_dim;
        const int label = target[oix * inner_dim + iix];
        int k;
        for (k = 0; k < nignores; k++) {
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

/*! <Tx = float32, Ty = float32, Device = CUDA> */

template <> void NLLLoss<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const float*            target,
    float*                  loss,
    int*                    flag,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * inner_dim;
    _NLLLoss
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nthreads, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, loss, flag
     );
}

/*! <Tx = float32, Ty = int64, Device = CUDA> */

template <> void NLLLoss<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const int64_t*          target,
    float*                  loss,
    int*                    flag,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * inner_dim;
    _NLLLoss
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nthreads, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, loss, flag
    );
}

/*! <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _NLLLossGrad(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               log_prob,
    const Ty*               target,
    Tx*                     dx,
    int*                    flag) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int oix = i / inner_dim;
        const int iix = i % inner_dim;
        const int label = target[oix * inner_dim + iix];
        int k;
        for (k = 0; k < nignores; k++)
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

/*! <Tx = float32, Ty = float32, Device = CUDA> */

template<> void NLLLossGrad<float, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const float*            target,
    float*                  dx,
    int*                    flag,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * inner_dim;
    _NLLLossGrad
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nthreads, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, dx, flag
    );
}

/*! <Tx = float32, Ty = int64, Device = CUDA> */

template<> void NLLLossGrad<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const float*            log_prob,
    const int64_t*          target,
    float*                  dx,
    int*                    flag,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * inner_dim;
    _NLLLossGrad
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nthreads, axis_dim, inner_dim, nignores,
        ignore, log_prob, target, dx, flag
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA