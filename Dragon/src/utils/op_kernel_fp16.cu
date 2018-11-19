#ifdef WITH_CUDA

#include <cmath>

#include "core/context_cuda.h"
#include "core/tensor.h"
#include "utils/cuda_device.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/cast.h"

namespace dragon {

namespace kernel {

/******************** activation.dropout ********************/

__global__ void _DropoutHalf(
    const int               count,
    const uint32_t          thresh,
    const half              scale,
    const half*             x,
    const uint32_t*         mask32,
    uint8_t*                mask8,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        mask8[idx] = (mask32[idx] > thresh);
        y[idx] = __hmul(__hmul(x[idx], scale),
            __float2half((float)mask8[idx]));
#endif
    }
}

template<> void Dropout<float16, CUDAContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float16*          x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float16*                y,
    CUDAContext*            ctx) {
    math::RandomUniform<uint32_t, CUDAContext>(
        count, float(0), float(UINT_MAX), mask32, ctx);
    auto thresh = static_cast<uint32_t>(UINT_MAX * prob);
    _DropoutHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 thresh, dragon_cast<half, float>(scale),
                     reinterpret_cast<const half*>(x),
                         mask32, mask8, reinterpret_cast<half*>(y));
}

template <typename Tm>
__global__ void _ApplyMaskHalf(
    const int               count,
    const half              scale,
    const half*             x,
    const Tm*               mask,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(__hmul(x[idx], scale),
            __float2half((float)mask[idx]));
#endif
    }
}

template <> void ApplyMask<float16, uint8_t, CUDAContext>(
    const int               count,
    const float             scale,
    const float16*          x,
    const uint8_t*          mask,
    float16*                y,
    CUDAContext*            ctx) {
    _ApplyMaskHalf<uint8_t>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dragon_cast<half, float>(scale),
                     reinterpret_cast<const half*>(x),
                         mask, reinterpret_cast<half*>(y));
}

/******************** activation.relu ********************/

template <typename T>
__global__ void _ReluHalf(
    const int               count,
    const half              slope,
    const half*             x,
    half*                   y) {
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hgt(x[idx], kZero) ?
            x[idx] : __hmul(x[idx], slope);
#endif
    }
}

template <typename T>
__global__ void _ReluHalf2(
    const int               count,
    const half2             slope,
    const half2*            x,
    half2*                  y) {
    const half2 kZero = __float2half2_rn(0.f);
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hbgt2(x[idx], kZero) ?
            x[idx] : __hmul2(x[idx], slope);
#endif
    }
}

template<> void Relu<float16, CUDAContext>(
    const int               count,
    const float             slope,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((count & 1) == 0) {
        _ReluHalf2<half2>
            << < CUDA_BLOCKS(count >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> > (count >> 1,
                     dragon_cast<half2, float>(slope),
                         reinterpret_cast<const half2*>(x),
                             reinterpret_cast<half2*>(y));
    } else {
        _ReluHalf<half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     dragon_cast<half, float>(slope),
                         reinterpret_cast<const half*>(x),
                             reinterpret_cast<half*>(y));
    }
}

/******************** arithmetic.affine ********************/

template <typename T>
__global__ void _AffineWithOBiasHalf(
    const int               count,
    const int               scale_dim,
    const int               inner_dim,
    const half*             x,
    const half*             alpha,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int scale_idx = (idx / inner_dim) % scale_dim;
        y[idx] = __hmul(alpha[scale_idx], x[idx]);
#endif
    }
}

template <typename T>
__global__ void _AffineWithBiasHalf(
    const int               count,
    const int               scale_dim,
    const int               inner_dim,
    const half*             x,
    const half*             alpha,
    const half*             beta,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int scale_idx = (idx / inner_dim) % scale_dim;
        y[idx] = __hadd(
            __hmul(alpha[scale_idx], x[idx]),
            beta[scale_idx]
        );
#endif
    }
}

template<> void Affine<float16, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float16*          x,
    const float16*          alpha,
    const float16*          beta,
    const float16*          beta_multiplier,
    float16*                y,
    CUDAContext*            ctx) {
    if (beta != nullptr) {
        _AffineWithBiasHalf<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     scale_dim, inner_dim,
                         reinterpret_cast<const half*>(x),
                             reinterpret_cast<const half*>(alpha),
                                 reinterpret_cast<const half*>(beta),
                                     reinterpret_cast<half*>(y));
    } else {
        _AffineWithOBiasHalf<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     scale_dim, inner_dim,
                         reinterpret_cast<const half*>(x),
                             reinterpret_cast<const half*>(alpha),
                                 reinterpret_cast<half*>(y));
    }
}

/******************** loss.nll_loss ********************/

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
                losses[idx] = flags[idx] = 0;
                break;
            }
        }
        if (k == num_ignores) {
            losses[idx] = __half2float(__hneg(
                log_prob[(oix * axis_dim + label) * inner_dim + iix]));
            flags[idx] = 1;
        }
#endif
    }
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(log_prob), labels,
                         ignores, num_ignores, losses, flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(log_prob), labels,
                         ignores, num_ignores, losses, flags);
}

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
            flags[idx] = 0;
        } else {
            dx[(oix * axis_dim + label) * inner_dim + iix] = __float2half(-1.);
            flags[idx] = 1;
        }
#endif
    }
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(log_prob), labels,
                         ignores, num_ignores,
                             reinterpret_cast<half*>(dx), flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(log_prob), labels,
                         ignores, num_ignores,
                             reinterpret_cast<half*>(dx), flags);
}

/******************** loss.sparse_softmax_cross_entropy ********************/

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
                losses[idx] = flags[idx] = 0;
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
            flags[idx] = 1;
        }
#endif
    }
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(prob), labels,
                         ignores, num_ignores, losses, flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(prob), labels,
                         ignores, num_ignores, losses, flags);
}

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
            flags[idx] = 0;
        } else {
            const int x_idx = (oix * axis_dim + label) * inner_dim + iix;
            dx[x_idx] = __hsub(dx[x_idx], __float2half(1.f));
            flags[idx] = 1;
        }
#endif
    }
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(prob), labels,
                         ignores, num_ignores,
                             reinterpret_cast<half*>(dx), flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     reinterpret_cast<const half*>(prob), labels,
                         ignores, num_ignores,
                             reinterpret_cast<half*>(dx), flags);

}

/******************** misc.astype ********************/

__global__ void _TypeHalf2Float(
    const int               count,
    const half*             a,
    float*                  b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = __half2float(a[idx]);
    }
}
__global__ void _TypeFloat2Half(
    const int               count,
    const float*            a,
    half*                   b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = __float2half(a[idx]);
    }
}

__global__ void _TypeHalf2Half(
    const int               count,
    const half*             a,
    half*                   b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = a[idx];
    }
}

#define DEFINE_TYPE_DISABLE_FP16(type) \
    template <> void TypeA2B<float16, type, CUDAContext>( \
        const int           count, \
        const float16*      a, \
        type*               b, \
        CUDAContext*        ctx) { \
        LOG(FATAL) << "CUDAContext has not implemented: float16 -> " \
                   << TypeMetaToString(TypeMeta::Make<type>()); \
    } \
    template <> void TypeA2B<type, float16, CUDAContext>( \
        const int           count, \
        const type*         a, \
        float16*            b, \
        CUDAContext*        ctx) { \
        LOG(FATAL) << "CUDAContext has not implemented: " \
                   << TypeMetaToString(TypeMeta::Make<type>()) << " -> float16"; \
    }

#define DEFINE_TYPE_ENABLE_FP16_FP32 \
    template <> void TypeA2B<float16, float, CUDAContext>( \
        const int           count, \
        const float16*      a, \
        float*              b, \
        CUDAContext*        ctx) { \
        _TypeHalf2Float \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >(count, \
                     reinterpret_cast<const half*>(a), b); \
    } \
    template <> void TypeA2B<float, float16, CUDAContext>( \
        const int           count, \
        const float*        a, \
        float16*            b, \
        CUDAContext*        ctx) { \
        _TypeFloat2Half \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >(count, \
                     a, reinterpret_cast<half*>(b)); \
    }

template <> void TypeA2B<float16, float16, CUDAContext>(
    const int               count,
    const float16*          a,
    float16*                b,
    CUDAContext*            ctx) {
    _TypeHalf2Half
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 reinterpret_cast<const half*>(a),
                     reinterpret_cast<half*>(b));
}

DEFINE_TYPE_ENABLE_FP16_FP32;
DEFINE_TYPE_DISABLE_FP16(double);
DEFINE_TYPE_DISABLE_FP16(int);
DEFINE_TYPE_DISABLE_FP16(int64_t);
DEFINE_TYPE_DISABLE_FP16(uint8_t);

/******************** misc.image_data ********************/

template <typename Tx, typename Ty>
__global__ void _ImageDataHalf_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / W / H) % C;
        const int n = idx / W / H / C;
        float raw_value = x[((n * H + h) * W + w) * C + c];
        if (mean_values) raw_value -= mean_values[c];
        if (std_values) raw_value /= std_values[c];
        y[idx] = __float2half(raw_value);
    }
}

template <typename Tx, typename Ty>
__global__ void _ImageDataHalf_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        float raw_value = x[idx];
        if (mean_values) raw_value -= mean_values[c];
        if (std_values) raw_value /= std_values[c];
        y[idx] = __float2half(raw_value);
    }
}

template <> void ImageData<float, float16, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const float*            x,
    float16*                y,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageDataHalf_NCHW<float, half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values,
                         x, reinterpret_cast<half*>(y));
    } else if (data_format == "NHWC") {
        _ImageDataHalf_NHWC<float, half> 
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values,
                         x, reinterpret_cast<half*>(y));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <> void ImageData<uint8_t, float16, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const uint8_t*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageDataHalf_NCHW<uint8_t, half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values,
                         x, reinterpret_cast<half*>(y));
    } else if (data_format == "NHWC") {
        _ImageDataHalf_NHWC<uint8_t, half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values,
                         x, reinterpret_cast<half*>(y));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** ndarray.concat ********************/

template <typename T>
__global__ void _ConcatHalf(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = x_concat_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int concat_idx = idx % tmp;
        const int y_idx = (outer_idx * y_concat_dim + concat_offset)
                                * inner_dim + concat_idx;
        y[y_idx] = x[idx];
    }
}

template <> void Concat<float16, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _ConcatHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 outer_dim, inner_dim,
                     x_concat_dim, y_concat_dim, concat_offset,
                         reinterpret_cast<const half*>(x),
                             reinterpret_cast<half*>(y));
}

template <typename T>
__global__ void _ConcatGradHalf(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = x_concat_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int concat_idx = idx % tmp;
        const int y_idx = (outer_idx * y_concat_dim + concat_offset)
                                * inner_dim + concat_idx;
        dx[idx] = dy[y_idx];
    }
}

template <> void ConcatGrad<float16, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    _ConcatGradHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 outer_dim, inner_dim,
                     x_concat_dim, y_concat_dim, concat_offset,
                         reinterpret_cast<const half*>(dy),
                             reinterpret_cast<half*>(dx));
}

/******************** ndarray.transpose ********************/

template <typename T>
__global__ void _TransposeHalf(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
       int x_idx = 0, y_idx = idx;
       for (int j = 0; j < ndim; ++j) {
           int k = order[j];
           x_idx += (y_idx / new_steps[j]) * old_steps[k];
           y_idx %= new_steps[j];
       }
       y[idx] = x[x_idx];
   }
}

template <> void Transpose<float16, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _TransposeHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 ndim, order, old_steps, new_steps,
                     reinterpret_cast<const half*>(x),
                         reinterpret_cast<half*>(y));
}

template <typename T>
__global__ void _TransposeGradHalf(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int x_idx = 0, y_idx = idx;
        for (int j = 0; j < ndim; ++j) {
            int k = order[j];
            x_idx += (y_idx / new_steps[j]) * old_steps[k];
            y_idx %= new_steps[j];
        }
        dx[x_idx] = dy[idx];
    }
}

template <> void TransposeGrad<float16, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    _TransposeGradHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 ndim, order, old_steps, new_steps,
                     reinterpret_cast<const half*>(dy),
                         reinterpret_cast<half*>(dx));
}

/******************** update.adam_update ********************/

__global__ void _AdamUpdateHalf(
    const int               count,
    const half              lr,
    const half              beta1,
    const half              beta2,
    const half              eps,
    half*                   g,
    half*                   m,
    half*                   v) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half gi = g[i];
        half kOne = __float2half(1.f);
        half mi = m[i] = __hadd(
            __hmul(m[i], beta1),
            __hmul(gi, __hsub(kOne, beta1))
        );
        half vi = v[i] = __hadd(
            __hmul(v[i], beta2),
            __hmul(gi, __hmul(gi, __hsub(kOne, beta2)))
        );
        g[i] = __hdiv(
            __hmul(lr, mi),
            __hadd(hsqrt(vi), eps)
        );
#endif
    }
}

template <> void AdamUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    float16*                g,
    float16*                m,
    float16*                v,
    CUDAContext*            ctx) {
    _AdamUpdateHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dragon_cast<half, float>(lr),
                     dragon_cast<half, float>(beta1),
                         dragon_cast<half, float>(beta2),
                             dragon_cast<half, float>(eps),
                                 reinterpret_cast<half*>(g),
                                     reinterpret_cast<half*>(m),
                                         reinterpret_cast<half*>(v));
}

/******************** update.nesterov_update ********************/

__global__ void _NesterovUpdateHalf(
    const int               count,
    const half              lr,
    const half              momentum,
    half*                   g,
    half*                   h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half hi = h[i];
        half hi_new = h[i] = __hadd(
            __hmul(momentum, hi),
            __hmul(lr, g[i])
        );
        half kOne = __float2half(1.f);
        g[i] = __hsub(
            __hmul(__hadd(kOne, momentum), hi_new),
            __hmul(momentum, hi)
        );
#endif
    }
}

__global__ void _NesterovUpdateHalf2(
    const int               count,
    const half2             lr,
    const half2             momentum,
    half2*                  g,
    half2*                  h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half2 hi = h[i];
        half2 hi_new = h[i] = __hadd2(
            __hmul2(momentum, hi),
            __hmul2(lr, g[i])
        );
        half2 kOne = __float2half2_rn(1.f);
        g[i] = __hsub2(
            __hmul2(__hadd2(kOne, momentum), hi_new),
            __hmul2(momentum, hi)
        );
#endif
    }
}

template <> void NesterovUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float16*                g,
    float16*                h,
    CUDAContext*            ctx) {
    if ((count & 1) == 0) {
        _NesterovUpdateHalf2
            << < CUDA_BLOCKS(count >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count >> 1,
                     dragon_cast<half2, float>(lr),
                         dragon_cast<half2, float>(momentum),
                             reinterpret_cast<half2*>(g),
                                 reinterpret_cast<half2*>(h));
    } else {
        _NesterovUpdateHalf
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     dragon_cast<half, float>(lr),
                         dragon_cast<half, float>(momentum),
                             reinterpret_cast<half*>(g),
                                 reinterpret_cast<half*>(h));
    }
}

/******************** update.rmsprop_update ********************/

__global__ void _RMSPropUpdateHalf(
    const int               count,
    const half              lr,
    const half              decay,
    const half              eps,
    half*                   g,
    half*                   h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half gi = g[i];
        half kOne = __float2half(1.f);
        half hi = h[i] = __hadd(
            __hmul(decay, h[i]),
            __hmul(__hmul(__hsub(kOne, decay), gi), gi)
        );
        g[i] = __hdiv(
            __hmul(lr, g[i]),
            __hadd(hsqrt(hi), eps)
        );
#endif
    }
}

template <> void RMSPropUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float16*                g,
    float16*                h,
    CUDAContext*            ctx) {
    _RMSPropUpdateHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dragon_cast<half, float>(lr),
                     dragon_cast<half, float>(decay),
                         dragon_cast<half, float>(eps),
                             reinterpret_cast<half*>(g),
                                 reinterpret_cast<half*>(h));
}

/******************** update.sgd_update ********************/

__global__ void _SGDUpdateHalf(
    const int               count,
    const half              lr,
    const half              momentum,
    half*                   g,
    half*                   h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half hi = h[i];
        g[i] = h[i] = __hadd(
            __hmul(momentum, hi),
            __hmul(lr, g[i])
        );
#endif
    }
}

__global__ void _SGDUpdateHalf2(
    const int               count,
    const half2             lr,
    const half2             momentum,
    half2*                  g,
    half2*                  h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half2 hi = h[i];
        g[i] = h[i] = __hadd2(
            __hmul2(momentum, hi),
            __hmul2(lr, g[i])
        );
#endif
    }
}

template <> void SGDUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float16*                g,
    float16*                h,
    CUDAContext*            ctx) {
    if ((count & 1) == 0) {
        _SGDUpdateHalf2
            << < CUDA_BLOCKS(count >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count >> 1,
                     dragon_cast<half2, float>(lr),
                         dragon_cast<half2, float>(momentum),
                             reinterpret_cast<half2*>(g),
                                 reinterpret_cast<half2*>(h));
    } else {
        _SGDUpdateHalf
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     dragon_cast<half, float>(lr),
                         dragon_cast<half, float>(momentum),
                             reinterpret_cast<half*>(g),
                                 reinterpret_cast<half*>(h));
    }
}

}    // namespace kernel

}    // namespace dragon

#endif // WITH_CUDA