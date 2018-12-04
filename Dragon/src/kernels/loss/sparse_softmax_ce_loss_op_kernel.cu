#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SparseSoftmaxCrossEntropy <Tx = float32, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SparseSoftmaxCrossEntropy(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               prob,
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
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropy<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            prob, labels, ignores, num_ignores,
                losses, flags);
}

/*! SparseSoftmaxCrossEntropy <Tx = float32, Ty = int64, Device = CUDA> */

template <> void SparseSoftmaxCrossEntropy<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropy<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            prob, labels, ignores, num_ignores,
                losses, flags);
}

/*! SparseSoftmaxCrossEntropy <Tx = float16, Ty = ?, Device = CUDA> */

template <typename Ty>
__global__ void _SparseSoftmaxCrossEntropyHalf(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const half*             prob,
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
            const half kMIN = __float2half(HFLT_MIN);
            half loss = __hneg(
                hlog(
                    __hgt(prob[(oix * axis_dim + label)
                            * inner_dim + iix], kMIN) ?
                                prob[(oix * axis_dim + label)
                                    * inner_dim + iix] : kMIN
                )
            );
            losses[idx] = __half2float(loss);
            flags[idx] = 1.f;
        }
#endif
    }
}

/*! SparseSoftmaxCrossEntropy <Tx = float16, Ty = float32, Device = CUDA> */

template <> void SparseSoftmaxCrossEntropy<float16, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyHalf<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(prob), labels,
                ignores, num_ignores, losses, flags);
}

/*! SparseSoftmaxCrossEntropy <Tx = float16, Ty = int64, Device = CUDA> */

template <> void SparseSoftmaxCrossEntropy<float16, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyHalf<int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(prob), labels,
                ignores, num_ignores, losses, flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float32, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _SparseSoftmaxCrossEntropyGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               prob,
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
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyGrad<float, float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            prob, labels, ignores, num_ignores,
                dx, flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float32, Ty = int64, Device = CUDA> */

template<> void SparseSoftmaxCrossEntropyGrad<float, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyGrad<float, int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            prob, labels, ignores, num_ignores,
                dx, flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float16, Ty = ?, Device = CUDA> */

template <typename Ty>
__global__ void _SparseSoftmaxCrossEntropyGradHalf(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const half*             prob,
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
            for (int c = 0; c < axis_dim; c++)
                dx[(oix * axis_dim + c) * inner_dim + iix]
                    = __float2half(0.f);
            flags[idx] = 0.f;
        } else {
            const int x_idx = (oix * axis_dim + label) * inner_dim + iix;
            dx[x_idx] = __hsub(dx[x_idx], __float2half(1.f));
            flags[idx] = 1.f;
        }
#endif
    }
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float16, Ty = float32, Device = CUDA> */

template<> void SparseSoftmaxCrossEntropyGrad<float16, float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float16*                dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyGradHalf<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(prob), labels,
                ignores, num_ignores,
                    reinterpret_cast<half*>(dx), flags);
}

/*! SparseSoftmaxCrossEntropyGrad <Tx = float16, Ty = int64, Device = CUDA> */

template<> void SparseSoftmaxCrossEntropyGrad<float16, int64_t, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          prob,
    const int64_t*          labels,
    const int*              ignores,
    const int               num_ignores,
    float16*                dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SparseSoftmaxCrossEntropyGradHalf<int64_t>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (num_preds, axis_dim, inner_dim,
            reinterpret_cast<const half*>(prob), labels,
                ignores, num_ignores,
                    reinterpret_cast<half*>(dx), flags);

}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA