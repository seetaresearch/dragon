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

template<typename T>
__global__ void _Dropout(
    const int               count,
    const uint32_t          thresh,
    const float             scale,
    const T*                x,
    const uint32_t*         mask32,
    uint8_t*                mask8,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        mask8[idx] = (mask32[idx] > thresh);
        y[idx] = x[idx] * mask8[idx] * scale;
    }
}

template<> void Dropout<float, CUDAContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float*            x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float*                  y,
    CUDAContext*            ctx) {
    math::RandomUniform<uint32_t, CUDAContext>(
        count, float(0), float(UINT_MAX), mask32, ctx);
    auto thresh = static_cast<uint32_t>(UINT_MAX * prob);
    _Dropout<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 thresh, scale, x, mask32, mask8, y);
}

template <typename Tx, typename Tm>
__global__ void _ApplyMask(
    const int               count,
    const float             scale,
    const Tx*               x,
    const Tm*               mask,
    Tx*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = x[idx] * mask[idx] * scale;
    }
}

template <> void ApplyMask<float, uint8_t, CUDAContext>(
    const int               count,
    const float             scale,
    const float*            x,
    const uint8_t*          mask,
    float*                  y,
    CUDAContext*            ctx) {
    _ApplyMask<float, uint8_t>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 scale, x, mask, y);
}

/******************** activation.prelu ********************/

template <typename T>
__global__ void _PRelu(
    const int               count,
    const int               channels,
    const int               dim,
    const T*                x,
    const T*                w,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = (x[idx] > 0) * x[idx] +
            (x[idx] < 0) * x[idx] * w[0];
    }
}

template <typename T>
__global__ void _PReluNCHW(
    const int               count,
    const int               channels,
    const int               dim,
    const T*                x,
    const T*                w,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = (idx / dim) % channels;
        y[idx] = (x[idx] > 0) * x[idx] +
            (x[idx] < 0) * x[idx] * w[c];
    }
}

template <typename T>
__global__ void _PReluNHWC(
    const int               count,
    const int               channels,
    const int               dim,
    const T*                x,
    const T*                w,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % channels;
        y[idx] = (x[idx] > 0) * x[idx] +
            (x[idx] < 0) * x[idx] * w[c];
    }
}

template<> void PRelu<float, CUDAContext>(const int count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            x,
    const float*            w,
    float*                  y,
    CUDAContext*            ctx) {
    if (channel_shared) {
        _PRelu<float> 
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     channels, dim, x, w, y);
    } else {
        if (data_format == "NCHW") {
            _PReluNCHW<float>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >(count,
                         channels, dim, x, w, y);
        } else if (data_format == "NHWC") {
            _PReluNHWC<float>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >(count,
                         channels, dim, x, w, y);
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

template <typename T>
__global__ void _PReluGrad(
    const int               count,
    const int               channels,
    const int               dim,
    const T*                dy,
    const T*                x,
    const T*                w,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = dy[idx] * (
            (x[idx] > 0) + (x[idx] <= 0) * w[0]
        );
    }
}

template <typename T>
__global__ void _PReluGradNCHW(
    const int               count,
    const int               channels,
    const int               dim,
    const T*                dy,
    const T*                x,
    const T*                w,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = (idx / dim) % channels;
        dx[idx] = dy[idx] * (
            (x[idx] > 0) + (x[idx] <= 0) * w[c]
        );
    }
}

template <typename T>
__global__ void _PReluGradNHWC(
    const int               count,
    const int               channels,
    const int               dim,
    const T*                dy,
    const T*                x,
    const T*                w,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % channels;
        dx[idx] = dy[idx] * ((x[idx] > 0) + (x[idx] <= 0) * w[c]);
    }
}

template<> void PReluGrad<float, CUDAContext>(
    const int               count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            dy,
    const float*            x,
    const float*            w,
    float*                  dx,
    CUDAContext*            ctx) {
    if (channel_shared) {
        _PReluGrad<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     channels, dim, dy, x, w, dx);
    } else {
        if (data_format == "NCHW") {
            _PReluGradNCHW<float>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >(count,
                         channels, dim, dy, x, w, dx);
        } else if (data_format == "NHWC") {
            _PReluGradNHWC<float>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >(count,
                         channels, dim, dy, x, w, dx);
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

template <typename T>
__global__ void _PReluWGradBcast(
    const int               count,
    const int               rows,
    const int               row_offset,
    const T*                dy,
    const T*                x,
    T*                      bcast_dw) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        bcast_dw[idx] = dy[idx] * x[idx] * (x[idx] <= 0);
        for (int n = 1; n < rows; n++) {
            const int cur_idx = idx + n * row_offset;
            bcast_dw[idx] +=
                dy[cur_idx] * x[cur_idx] * (x[cur_idx] <= 0);
        }
    }
}

template<> void PReluWGrad<float, CUDAContext>(
    const int               rows,
    const int               row_offset,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            dy,
    const float*            x,
    const float*            multiplier,
    float*                  bcast_dw,
    float*                  dw,
    CUDAContext*            ctx) {
    const int cdim = channels * dim;
    _PReluWGradBcast<float> 
        << < CUDA_BLOCKS(cdim), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 cdim, rows, row_offset, dy, x, bcast_dw);
    if (channel_shared) {
        float w_sum;
        math::Dot<float, CUDAContext>(channels * dim,
            bcast_dw, multiplier, &w_sum, ctx);
        math::AddScalar<float, CUDAContext>(1, w_sum, dw, ctx);
    } else {
        if (data_format == "NCHW") {
            math::Gemv<float, CUDAContext>(
                CblasNoTrans, channels, dim,
                1.0, bcast_dw, multiplier, 1.0, dw, ctx);
        } else if (data_format == "NHWC") {
            math::Gemv<float, CUDAContext>(
                CblasTrans, dim, channels,
                1.0, bcast_dw, multiplier, 1.0, dw, ctx);
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

/******************** activation.elu ********************/

template <typename T>
__global__ void _Elu(
    const int               count,
    const T*                x,
    const float             alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = x[idx] > 0 ? x[idx] :
            alpha * (exp(x[idx]) - 1);
    }
}

template<> void Elu<float, CUDAContext>(
    const int               count,
    const float             alpha,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Elu<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x, alpha, y);
}

template <typename T>
__global__ void _EluGrad(
    const int               count,
    const float             alpha,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = dy[idx] * (
            (y[idx] > 0) + (alpha + y[idx]) * (y[idx] <= 0)
        );
    }
}

template<> void EluGrad<float, CUDAContext>(
    const int               count,
    const float             alpha,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _EluGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, alpha, dy, y, dx);
}

/******************** activation.relu ********************/

template <typename T>
__global__ void _Relu(
    const int               count,
    const float             slope,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = x[idx] > 0 ? x[idx] : x[idx] * slope;
    }
}

template<> void Relu<float, CUDAContext>(
    const int               count,
    const float             slope,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Relu<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, slope, x, y);
}

template <typename T>
__global__ void _ReluGrad(
    const int               count,
    const float             slope,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = dy[idx] * (
            (y[idx] > 0) + slope * (y[idx] <= 0)
        );
    }
}

template<> void ReluGrad<float, CUDAContext>(
    const int               count,
    const float             slope,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _ReluGrad<float> 
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, slope, dy, y, dx);
}

/******************** activation.selu ********************/

template <typename T>
__global__ void _SElu(
    const int               count,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = x[idx] > 0 ? 1.0507 * x[idx] :
            1.7581 * (exp(x[idx]) - 1);
    }
}

template<> void SElu<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _SElu<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x, y);
}

template <typename T>
__global__ void _SEluGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = y[idx] > 0 ? 1.0507 * dy[idx] :
            (1.7581 + y[idx]) * dy[idx];
    }
}

template<> void SEluGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _SEluGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, dy, y, dx);
}

/******************** activation.sigmoid ********************/

template <typename T>
__device__ T _SigmoidUnit(const T x) {
    return T(1) / (T(1) + exp(-x)); 
}

template <typename T>
__global__ void _Sigmoid(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = _SigmoidUnit<T>(x[idx]);
    }
}

template<> void Sigmoid<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Sigmoid<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x, y);
}

template <typename T>
__global__ void _SigmoidGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx[idx] = dy[idx] * y[idx] * (1 - y[idx]);
    }
}

template<> void SigmoidGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _SigmoidGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, dy, y, dx);
}

/******************** activation.softmax ********************/

template <typename T>
__global__ void _SoftmaxMaxClass(
    const int               outer_dim,
    const int               classes,
    const int               inner_dim,
    const T*                x,
    T*                      scale) {
    CUDA_1D_KERNEL_LOOP(idx, outer_dim * inner_dim) {
        int o_idx = idx / inner_dim;
        int i_idx = idx % inner_dim;
        T max_val = -FLT_MAX;
        for (int c = 0; c < classes; c++)
            max_val = max(
                x[(o_idx * classes + c) * inner_dim + i_idx], max_val
            );
        scale[idx] = max_val;
    }
}

template <typename T>
__global__ void _SoftmaxSubtract(
    const int               count,
    const int               classes,
    const int               inner_dim,
    const T*                scale,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int o_idx = idx / inner_dim / classes;
        int i_idx = idx % inner_dim;
        y[idx] -= scale[o_idx * inner_dim + i_idx];
    }
}

template <typename T>
__global__ void _SoftmaxExp(
    const int               count,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = exp(y[idx]);
    }
}

template <typename T>
__global__ void _SoftmaxSumClass(
    const int               outer_dim,
    const int               classes,
    const int               inner_dim,
    const T*                y,
    T* scale) {
    CUDA_1D_KERNEL_LOOP(idx, outer_dim * inner_dim) {
        int o_idx = idx / inner_dim;
        int i_idx = idx % inner_dim;
        T sum = 0;
        for (int c = 0; c < classes; c++)
            sum += y[(o_idx * classes + c) * inner_dim + i_idx];
        scale[idx] = sum;
    }
}

template <typename T>
 __global__ void _SoftmaxDiv(
     const int              count,
     const int              classes,
     const int              inner_dim,
     const T*               scale,
     T*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int o_idx = idx / inner_dim / classes;
        int i_idx = idx % inner_dim;
        y[idx] /= scale[o_idx * inner_dim + i_idx];
    }
}

template<> void Softmax<float, CUDAContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            sum_multiplier,
    const float*            x,
    float*                  scale,
    float*                  y,
    CUDAContext*            ctx) {
    const int num_preds = inner_dim * outer_dim;
    _SoftmaxMaxClass<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 outer_dim, classes, inner_dim, x, scale);
    _SoftmaxSubtract<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, classes, inner_dim, scale, y);
    _SoftmaxExp<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, y);
    _SoftmaxSumClass<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 outer_dim, classes, inner_dim, y, scale);
    _SoftmaxDiv<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                count, classes, inner_dim, scale, y);
}

template <typename T>
__global__ void _SoftmaxDot(
    const int               outer_dim,
    const int               classes,
    const int               inner_dim,
    const T*                dy,
    const T*                y,
    T*                      scale) {
    CUDA_1D_KERNEL_LOOP(idx, outer_dim * inner_dim) {
        int o_idx = idx / inner_dim;
        int i_idx = idx % inner_dim;
        T dot = 0;
        for (int c = 0; c < classes; c++)
            dot += (
                y[(o_idx * classes + c) * inner_dim + i_idx] *
                    dy[(o_idx * classes + c) * inner_dim + i_idx]
            );
        scale[idx] = dot;
    }
}

template<> void SoftmaxGrad<float, CUDAContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            sum_multiplier,
    const float*            dy,
    const float*            y,
    float*                  scale,
    float*                  dx,
    CUDAContext*            ctx) {
    const int num_preds = inner_dim * outer_dim;
    _SoftmaxDot<float> 
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 outer_dim, classes, inner_dim, dy, y, scale);
    _SoftmaxSubtract<float> 
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, classes,inner_dim, scale, dx);
    math::Mul<float, CUDAContext>(count, dx, y, dx, ctx);
}

/******************** activation.tanh ********************/

template <typename T>
__global__ void _Tanh(
    const int               count,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        y[i] = tanh(x[i]);
    }
}

template<> void Tanh<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Tanh<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x, y);
}

template <typename T>
__global__ void _TanhGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
}

template<> void TanhGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _TanhGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, dy, y, dx);
}

/******************** arithmetic.scale ********************/

template <typename T>
__global__ void _AffineWithOBias(
    const int               count,
    const int               scale_dim,
    const int               inner_dim,
    const T*                x,
    const T*                alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int scale_idx = (idx / inner_dim) % scale_dim;
         y[idx] = alpha[scale_idx] * x[idx];
    }
}

template <typename T>
__global__ void _AffineWithBias(
    const int               count,
    const int               scale_dim,
    const int               inner_dim,
    const T*                x,
    const T*                alpha,
    const T*                beta,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int scale_idx = (idx / inner_dim) % scale_dim;
        y[idx] = alpha[scale_idx] * x[idx] + beta[scale_idx];
    }
}

template<> void Affine<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float*            x,
    const float*            alpha,
    const float*            beta,
    const float*            beta_multiplier,
    float*                  y,
    CUDAContext*            ctx) {
    if (beta != nullptr) {
        _AffineWithBias<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
                     count, scale_dim, inner_dim,
                         x, alpha, beta, y);
    } else {
        _AffineWithOBias<float>
            << <CUDA_BLOCKS(count), CUDA_THREADS,
                0, ctx->cuda_stream() >> >(
                    count, scale_dim, inner_dim,
                        x, alpha, y);
    }
}

template <> void AffineGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float*            dy,
    const float*            alpha,
    float*                  dx,
    CUDAContext*            ctx) {
    _AffineWithOBias<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, scale_dim, inner_dim,
                     dy, alpha, dx);
}

/******************** arithmetic.clip ********************/

template <typename T>
__global__ void _Clip(
    const int               count,
    const T                 low,
    const T                 high,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = max(low, min(x[idx], high));
    }
}

template <> void Clip<float, CUDAContext>(
    const int               count,
    const float             low,
    const float             high,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Clip<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, low, high, x, y);
}

template <typename T>
__global__ void _ClipGrad(
    const int               count,
    const T                 low,
    const T                 high,
    const T*                x,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const T xi = x[idx];
        dx[idx] = (xi < low || xi > high) ? 0 : dy[idx];
    }
}

template <> void ClipGrad<float, CUDAContext>(
    const int               count,
    const float             low,
    const float             high,
    const float*            x,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ClipGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 low, high, x, dy, dx);
}

/******************** arithmetic.maximum ********************/

template <typename T>
__global__ void _MaximumE(
    const int               count,
    const T*                x1,
    const T*                x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = max(x1[idx], x2[idx]);
    }
}

template <> void MaximumE<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    float*                  y,
    CUDAContext*            ctx) {
    _MaximumE<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
              0, ctx->cuda_stream() >> >(count, x1, x2, y);
}

template <typename T>
__global__ void _MaximumB(
    const int               count,
    const T*                x1,
    const T                 x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = max(x1[idx], x2);
    }
}

template <> void MaximumB<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    float*                  y,
    CUDAContext*            ctx) {
    _MaximumB<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x1, x2, y);
}

template <typename T>
__global__ void _MaximumEGrad(
    const int               count,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const bool dy_to_dx1 = x1[idx] > x2[idx];
        dx1[idx] = dy_to_dx1 ? dy[idx] : 0;
        dx2[idx] = dy_to_dx1 ? 0 : dy[idx];
    }
}

template <> void MaximumEGrad<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    const float*            dy,
    float*                  dx1,
    float*                  dx2,
    CUDAContext*            ctx) {
    _MaximumEGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x1, x2, dy, dx1, dx2);
}

template <typename T>
__global__ void _MaximumBGrad(
    const int               count,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx1[idx] = (x1[idx] > x2) ? dy[idx] : 0;
    }
}

template <> void MaximumBGrad<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    const float*            dy,
    float*                  dx1,
 /* float*                  dx2, */
    CUDAContext*            ctx) {
    _MaximumBGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x1, x2, dy, dx1);
}

/******************** arithmetic.minimum ********************/

template <typename T>
__global__ void _MinimumE(
    const int               count,
    const T*                x1,
    const T*                x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = min(x1[idx], x2[idx]);
    }
}

template <> void MinimumE<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    float*                  y,
    CUDAContext*            ctx) {
    _MinimumE<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
              0, ctx->cuda_stream() >> >(count, x1, x2, y);
}

template <typename T>
__global__ void _MinimumB(
    const int               count,
    const T*                x1,
    const T                 x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = min(x1[idx], x2);
    }
}

template <> void MinimumB<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    float*                  y,
    CUDAContext*            ctx) {
    _MinimumB<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x1, x2, y);
}

template <typename T>
__global__ void _MinimumEGrad(
    const int               count,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const bool dy_to_dx1 = x1[idx] < x2[idx];
        dx1[idx] = dy_to_dx1 ? dy[idx] : 0;
        dx2[idx] = dy_to_dx1 ? 0 : dy[idx];
    }
}

template <> void MinimumEGrad<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    const float*            dy,
    float*                  dx1,
    float*                  dx2,
    CUDAContext*            ctx) {
    _MinimumEGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x1, x2, dy, dx1, dx2);
}

template <typename T>
__global__ void _MinimumBGrad(
    const int               count,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        dx1[idx] = (x1[idx] < x2) ? dy[idx] : 0;
    }
}

template <> void MinimumBGrad<float, CUDAContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    const float*            dy,
    float*                  dx1,
 /* float*                  dx2, */
    CUDAContext*            ctx) {
    _MinimumBGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, x1, x2, dy, dx1);
}

/******************** control_flow.compare ********************/

template <typename T>
__global__ void _Equal(
    const int               count,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = fabs(a[idx] - b[idx]) < FLT_EPSILON ? 1.0 : 0.0;
    }
}

template <> void Equal<float, CUDAContext>(
    const int               count,
    const float*            a,
    const float*            b,
    float*                  y,
    CUDAContext*            ctx) {
    _Equal<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, a, b, y);
}

/******************** loss.l1_loss ********************/

template <typename T>
__global__ void _AbsGrad(
    const int               count,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
       const T val = dy[idx];
       //  val > 0: 1 | val == 0: 0 | val < 0: -1
       dx[idx] = (val > T(0)) - (val < T(0));
    }
}

template<> void AbsGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _AbsGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, dy, dx);
}

/******************** loss.nll_loss ********************/

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                    log_prob, labels, ignores,
                        num_ignores, losses, flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     log_prob, labels, ignores,
                        num_ignores, losses, flags);
}

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
             0, ctx->cuda_stream() >> >(
                num_preds, axis_dim, inner_dim,
                    log_prob, labels, ignores,
                        num_ignores, dx, flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                    log_prob, labels, ignores,
                        num_ignores, dx, flags);
}

/******************** loss.sigmoid_cross_entropy ********************/

template <typename T>
__global__ void _SigmoidCrossEntropy(
    const int               count,
    const T*                logits,
    const T*                targets,
    T*                      losses,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (targets[idx] < 0) {
            losses[idx] = flags[idx] = 0;
        } else {
            losses[idx] = log(1 +
                exp(logits[idx] - 2 * logits[idx] * (logits[idx] >= 0))
            ) + logits[idx] * ((logits[idx] >= 0) - targets[idx]);
            flags[idx] = 1;
        }
    }
}

template <> void SigmoidCrossEntropy<float, CUDAContext>(
    const int               count,
    const float*            logits,
    const float*            targets,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    _SigmoidCrossEntropy<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, logits, targets, losses, flags);
}

template <typename T>
__global__ void _SigmoidCrossEntropyGrad(
    const int               count,
    const T*                logits,
    const T*                targets,
    T*                      dlogits,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (targets[idx] < 0) {
            dlogits[idx] = flags[idx] = 0;
        } else {
            dlogits[idx] = 1 / (1 + exp(-logits[idx])) - targets[idx];
            flags[idx] = 1;
        }
    }
}

template <> void SigmoidCrossEntropyGrad<float, CUDAContext>(
    const int               count,
    const float*            logits,
    const float*            targets,
    float*                  dlogits,
    float*                  flags,
    CUDAContext*            ctx) {
    _SigmoidCrossEntropyGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, logits, targets, dlogits, flags);
}

/******************** loss.sigmoid_focal_loss ********************/

template <typename T>
__global__ void _SigmoidFocalLoss(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                logits,
    const T*                targets,
    T*                      losses,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int iix = idx % inner_dim;
        const int aix = (idx / inner_dim) % axis_dim;
        const int oix = idx / inner_dim / axis_dim;
        const int t = targets[oix * inner_dim + iix];
        //  ``0`` is reserved for targets if neg id is zero
        //  use ``aix + 1`` to match the targets
        T c1 = (t == (aix + (neg_id ? 0 : 1)));
        T c2 = (t != -1) & (t != (aix + (neg_id ? 0 : 1)));
        T p = 1 / (1 + exp(-logits[idx]));  //  logit -> prob

        // (1 - p)^{gamma} * log(p)
        T pos_term = pow(1 - p, gamma) * log(max(p, FLT_MIN));

        // p^{gamma} * log(1 - p)
        T neg_term = pow(p, gamma) * (
            -logits[idx] * (logits[idx] >= 0) - log(
                1 + exp(logits[idx] - 2 * logits[idx] * (logits[idx] >= 0)))
       );

        losses[idx] = 0.0;
        losses[idx] += -c1 * pos_term * pos_alpha;
        losses[idx] += -c2 * neg_term * neg_alpha;
        flags[idx] = c1;
    }
}

template <> void SigmoidFocalLoss<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const float*            targets,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    TIndex count = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLoss<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, axis_dim, inner_dim,
                     pos_alpha, neg_alpha, gamma, neg_id,
                         logits, targets, losses, flags);
}

template <typename T>
__global__ void _SigmoidFocalLossGradient(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                logits,
    const T*                targets,
    T*                      dlogits,
    T*                      flags) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int iix = idx % inner_dim;
        const int aix = (idx / inner_dim) % axis_dim;
        const int oix = idx / inner_dim / axis_dim;
        const int t = targets[oix * inner_dim + iix];
        //  ``0`` is reserved for targets if neg id is zero
        //  use ``aix + 1`` to match the targets
        T c1 = (t == (aix + (neg_id ? 0 : 1)));
        T c2 = (t != -1) & (t != (aix + (neg_id ? 0 : 1)));
        T p = 1 / (1 + exp(-logits[idx]));  //  logit -> prob

        // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
        T pos_term = pow((1 - p), gamma) * (
            1 - p - p * gamma * log(max(p, FLT_MIN))
        );

        // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
        T neg_term = pow(p, gamma) * (
            (-logits[idx] * (logits[idx] >= 0) - log(
                1 + exp(logits[idx] - 2 * logits[idx] * (logits[idx] >= 0)))
            ) * (1 - p) * gamma - p
        );

        dlogits[idx] = 0.0;
        dlogits[idx] += -c1 * pos_term * pos_alpha;
        dlogits[idx] += -c2 * neg_term * neg_alpha;
        flags[idx] = c1;
    }
}

template <> void SigmoidFocalLossGradient<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const float*            targets,
    float*                  dlogits,
    float*                  flags,
    CUDAContext*            ctx) {
    TIndex count = outer_dim * axis_dim * inner_dim;
    _SigmoidFocalLossGradient<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, axis_dim, inner_dim,
                     pos_alpha, neg_alpha, gamma, neg_id,
                         logits, targets, dlogits, flags);
}

/******************** loss.smooth_l1_loss ********************/

template <typename T>
__global__ void _SmoothL1(
    const int               count,
    const float             beta,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const T val = x[idx];
        const T abs_val = abs(val);
        if (abs_val < beta) y[idx] = 0.5 * val * val / beta;
        else y[idx] = abs_val - 0.5 * beta;
    }
}

template<> void SmoothL1<float, CUDAContext>(
    const int               count,
    const float             beta,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _SmoothL1<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, beta, x, y);
}

template <typename T>
__global__ void _SmoothL1Grad(
    const int               count,
    const float             beta,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const T val = dy[idx];
        const T abs_val = abs(val);
        if (abs_val < beta) dx[idx] = val / beta;
        //  val > 0: 1 | val == 0: 0 | val < 0: -1
        else dx[idx] = (val > T(0)) - (val < T(0));
    }
}

template<> void SmoothL1Grad<float, CUDAContext>(
    const int               count,
    const float             beta,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _SmoothL1Grad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, beta, dy, dx);
}

/******************** loss.softmax_cross_entropy ********************/

template <typename T>
__global__ void _SoftmaxCrossEntropy(
    const int               count,
    const T*                prob,
    const T*                target,
    T*                      loss) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        loss[idx] = -target[idx] * log(max(prob[idx], FLT_MIN));
    }
}

template <> void SoftmaxCrossEntropy<float, CUDAContext>(
    const int               count,
    const float*            prob,
    const float*            target,
    float*                  loss,
    CUDAContext*            ctx) {
    _SoftmaxCrossEntropy<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, prob, target, loss);
}

/******************** loss.softmax_focal_loss ********************/

template <typename T>
__global__ void _SoftmaxFocalLoss(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                prob,
    const T*                labels,
    const int*              ignores,
    const int               num_ignores,
    T*                      losses,
    T*                      flags) {
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
            const int t = (oix * axis_dim + label) * inner_dim + iix;
            T scale = pow(1.f - prob[t], gamma);
            scale = label > neg_id ?
                pos_alpha * scale : neg_alpha * scale;
            losses[idx] = -scale * log(max(prob[t], FLT_MIN));
            flags[idx] = label > neg_id ? 1 : 0;
        }
    }
}

template <> void SoftmaxFocalLoss<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLoss<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     pos_alpha, neg_alpha, gamma, neg_id,
                         prob, labels, ignores, num_ignores,
                             losses, flags);
}

template <typename T>
__global__ void _SoftmaxFocalLossGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                prob,
    const T*                labels,
    const int*              ignores,
    const int               num_ignores,
    T*                      dx,
    T*                      flags) {
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
            const int t = (oix * axis_dim + label) * inner_dim + iix;
            T onemp = 1. - prob[t];
            //  unstable if gamma is 0
            T grad = -gamma * pow(onemp, gamma - 1)
                            * log(max(prob[t], FLT_MIN))
                            * prob[t] + pow(onemp, gamma);
            grad = label > neg_id ?
                pos_alpha * grad : neg_alpha * grad;
            for (int c = 0; c < axis_dim; c++) {
                const int i = (oix * axis_dim + c) * inner_dim + iix;
                if (c == label) {
                    dx[i] = grad * (prob[t] - 1);
                } else {
                    dx[i] = grad * prob[i];
                }
            }
            flags[idx] = label > neg_id ? 1 : 0;
        }
    }
}

template<> void SoftmaxFocalLossGrad<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const float*            labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  dx,
    float*                  flags,
    CUDAContext*            ctx) {
    const int num_preds = outer_dim * inner_dim;
    _SoftmaxFocalLossGrad<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     pos_alpha, neg_alpha, gamma, neg_id,
                         prob, labels, ignores, num_ignores,
                             dx, flags);
}

/******************** loss.sparse_softmax_cross_entropy ********************/

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     prob, labels, ignores, num_ignores,
                         losses, flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     prob, labels, ignores, num_ignores,
                         losses, flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     prob, labels, ignores, num_ignores,
                         dx, flags);
}

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
             0, ctx->cuda_stream() >> >(
                 num_preds, axis_dim, inner_dim,
                     prob, labels, ignores, num_ignores,
                         dx, flags);
}

/******************** misc.astype ********************/

template <typename Ta, typename Tb>
__global__ void _TypeA2B(
    const int               count,
    const Ta*               a,
    Tb*                     b) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        b[idx] = a[idx];
    }
}

#define DEFINE_TYPE_A2B(type_a, type_b) \
    template <> void TypeA2B<type_a, type_b, CUDAContext>( \
        const int           count, \
        const               type_a* a, \
        type_b*             b, \
        CUDAContext*        ctx) { \
        _TypeA2B<type_a, type_b> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >(count, a, b); \
    }

#define DEFINE_TYPE_A2ALL(type_a) \
    DEFINE_TYPE_A2B(type_a, float); \
    DEFINE_TYPE_A2B(type_a, double); \
    DEFINE_TYPE_A2B(type_a, int); \
    DEFINE_TYPE_A2B(type_a, int64_t); \
    DEFINE_TYPE_A2B(type_a, uint8_t);

DEFINE_TYPE_A2ALL(float);
DEFINE_TYPE_A2ALL(double);
DEFINE_TYPE_A2ALL(int);
DEFINE_TYPE_A2ALL(int64_t);
DEFINE_TYPE_A2ALL(uint8_t);

/******************** misc.image_data ********************/

template <typename Tx, typename Ty>
__global__ void _ImageData_NCHW(
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
        Ty raw_value = x[((n * H + h) * W + w) * C + c];
        if (mean_values != nullptr) raw_value -= mean_values[c];
        if (std_values != nullptr) raw_value /= std_values[c];
        y[idx] = raw_value;
    }
}

template <typename Tx, typename Ty>
__global__ void _ImageData_NHWC(
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
        Ty raw_value = x[idx];
        if (mean_values != nullptr) raw_value -= mean_values[c];
        if (std_values != nullptr) raw_value /= std_values[c];
        y[idx] = raw_value;
    }
}

template <> void ImageData<float, float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<float, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<float, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <> void ImageData<uint8_t, float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const uint8_t*          x,
    float*                  y,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<uint8_t, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<uint8_t, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** ndarray.arange ********************/

template <typename T>
__global__ void _Arange(
    const int               count,
    const int               start,
    const int               step,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = start + idx * step;
    }
}

template<> void Arange<float, CUDAContext>(
    const int               count,
    const int               start,
    const int               step,
    float*                  y,
    CUDAContext*            ctx) {
    _Arange<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, start, step, y);
}

template<> void Arange<int, CUDAContext>(
    const int               count,
    const int               start,
    const int               step,
    int*                    y,
    CUDAContext*            ctx) {
    _Arange<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, start, step, y);
}

/******************** ndarray.argreduce ********************/

template <typename T>
__global__ void _Argmax(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 neg_bound,
    const T*                x,
    int64_t*                indices) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int max_idx = -1; T max_val = neg_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        indices[idx] = max_idx;
    }
}

template <typename T>
__global__ void _Argmax_v2(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 neg_bound,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int max_idx = -1; T max_val = neg_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        indices[idx] = max_idx;
        values[idx] = max_val;
    }
}

template<> void Argmax<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values,
    CUDAContext*            ctx) {
    CHECK_EQ(top_k, 1) << "top_k > 1 is not supported with CUDA";
    if (values == nullptr) {
        _Argmax<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     axis_dim, inner_dim, -FLT_MAX,
                         x, indices);
    } else {
        _Argmax_v2<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     axis_dim, inner_dim, -FLT_MAX,
                         x, indices, values);
    }
}

template <typename T>
__global__ void _Argmin(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 pos_bound,
    const T*                x,
    int64_t*                indices) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int min_idx = -1; T min_val = pos_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        indices[idx] = min_idx;
    }
}

template <typename T>
__global__ void _Argmin_v2(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 pos_bound,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int min_idx = -1; T min_val = pos_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        indices[idx] = min_idx;
        values[idx] = min_val;
    }
}

template<> void Argmin<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values,
    CUDAContext*            ctx) {
    CHECK_EQ(top_k, 1) << "top_k > 1 is not supported with CUDA";
    if (values == nullptr) {
        _Argmin<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     axis_dim, inner_dim, FLT_MAX,
                         x, indices);
    } else {
        _Argmin_v2<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     axis_dim, inner_dim, FLT_MAX,
                         x, indices, values);
    }
}

/******************** ndarray.gather ********************/

template <typename T>
__global__ void _CanonicalAxis(
    const int               count,
    const int               dim,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (y[idx] < 0) y[idx] += dim;
    }
}

template <> void CanonicalAxis<int, CUDAContext>(
    const int               count,
    const int               dim,
    int*                    y,
    CUDAContext*            ctx) {
    _CanonicalAxis<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count, dim, y);
}

template <typename T>
__global__ void _Gather(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int outer_idx = idx / inner_dim / y_slice_dim;
        const int slice_idx = idx % inner_dim;
        const int y_idx_offset = (idx / inner_dim) % y_slice_dim;
        const int x_idx_offset = indices[y_idx_offset];
        const int x_idx = (outer_idx * x_slice_dim + x_idx_offset)
                                     * inner_dim + slice_idx;
        y[idx] = x[x_idx];
    }
}

template <> void Gather<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Gather<float>
        << <CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >> >(count,
                outer_dim, inner_dim,
                    x_slice_dim, y_slice_dim,
                        indices, x, y);
}

template <> void Gather<int, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              x,
    int*                    y,
    CUDAContext*            ctx) {
    _Gather<int>
        << <CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >> >(count,
                outer_dim, inner_dim,
                    x_slice_dim, y_slice_dim,
                        indices, x, y);
}

template <typename T>
__global__ void _GatherGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int outer_idx = idx / inner_dim / y_slice_dim;
        const int slice_idx = idx % inner_dim;
        const int y_idx_offset = (idx / inner_dim) % y_slice_dim;
        const int x_idx_offset = indices[y_idx_offset];
        const int x_idx = (outer_idx * x_slice_dim + x_idx_offset)
                                     * inner_dim + slice_idx;
        atomicAdd(dx + x_idx, dy[idx]);
    }
}

template <> void GatherGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _GatherGrad<float>
        << <CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >> >(count,
                outer_dim, inner_dim,
                    x_slice_dim, y_slice_dim,
                        indices, dy, dx);
}

template <> void GatherGrad<int, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              dy,
    int*                    dx,
    CUDAContext*            ctx) {
    _GatherGrad<int>
        << <CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >> >(count,
                outer_dim, inner_dim,
                    x_slice_dim, y_slice_dim,
                        indices, dy, dx);
}

/******************** ndarray.concat ********************/

template <typename T>
__global__ void _Concat(
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

template <> void Concat<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Concat<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 outer_dim, inner_dim,
                     x_concat_dim, y_concat_dim,
                         concat_offset, x, y);
}

template <typename T>
__global__ void _ConcatGrad(
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

template <> void ConcatGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ConcatGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 outer_dim, inner_dim,
                     x_concat_dim, y_concat_dim,
                         concat_offset, dy, dx);
}

/******************** ndarray.crop ********************/

template<typename T>
__global__ void _Crop1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        y[idx] = x[(o * dim + ex_d + start) * inner_dim + i];
    }
}

template<> void Crop1D<int, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int*              x,
    int*                    y,
    CUDAContext*            ctx) {
    _Crop1D<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, start, x, y);
}

template<> void Crop1D<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Crop1D<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, start, x, y);
}

template<typename T>
__global__ void _Crop1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int d = (idx / inner_dim) % dim;
        const int o = idx / inner_dim / dim;
        dx[idx] = (d < start || d >= end) ? 0 :
            dy[(o * ex_dim + d - start) * inner_dim + i];
    }
}

template<> void Crop1DGrad<int, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const int*              dy,
    int*                    dx,
    CUDAContext*            ctx) {
    _Crop1DGrad<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, start, end, dy, dx);
}

template<> void Crop1DGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _Crop1DGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, start, end, dy, dx);
}

/******************** ndarray.pad ********************/

template <typename T>
__global__ void _ConstPad1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T                 value,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        const int d = ex_d - pad_l;
        y[idx] = (d < 0 || d >= dim) ? value :
            x[(o * dim + d) * inner_dim + i];
    }
}

template <> void ConstPad1D<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float             value,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _ConstPad1D<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, pad_l, value, x, y);
}

template <typename T>
__global__ void _ReflectPad1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        int d = ex_d - pad_l;
        d = max(d, -d);
        d = min(d, 2 * dim - d - 2);
        y[idx] = x[(o * dim + d) * inner_dim + i];
    }
}

template <> void ReflectPad1D<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _ReflectPad1D<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, pad_l, x, y);
}

template <typename T>
__global__ void _EdgePad1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        const int d = min(dim - 1, max(ex_d - pad_l, 0));
        y[idx] = x[(o * dim + d) * inner_dim + i];
    }
}

template <> void EdgePad1D<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _EdgePad1D<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, pad_l, x, y);
}

template <typename T>
__global__ void _ConstPad1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % dim + pad_l;
        const int o = idx / inner_dim / dim;
        dx[idx] = dy[(o * ex_dim + ex_d) * inner_dim + i];
    }
}

template <> void ConstPad1DGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ConstPad1DGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, pad_l, dy, dx);
}

template <typename T>
__global__ void _ReflectPad1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        int d = ex_d - pad_l;
        d = max(d, -d);
        d = min(d, 2 * dim - d - 2);
        atomicAdd(&dx[(o * dim + d) * inner_dim + i], dy[idx]);
    }
}

template <> void ReflectPad1DGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ReflectPad1DGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, pad_l, dy, dx);
}

template <typename T>
__global__ void _EdgePad1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        const int d = min(dim - 1, max(ex_d - pad_l, 0));
        atomicAdd(&dx[(o * dim + d) * inner_dim + i], dy[idx]);
    }
}

template <> void EdgePad1DGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _EdgePad1DGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 dim, ex_dim, inner_dim, pad_l, dy, dx);
}

/******************** ndarray.one_hot ********************/

template <typename T>
__global__ void _OneHot(
    const int               count,
    const int               depth,
    const int               on_value,
    const float*            x,
    float*                  y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int val = x[idx];
        y[idx * depth + val] = on_value;
    }
}

template <> void OneHot<float, CUDAContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _OneHot<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 depth, on_value, x, y);
}

/******************** ndarray.reduce ********************/

template <typename T>
__global__ void _Sum(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T*                x,
    float*                  y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T sum_val = 0.0;
        const int offset = (idx / inner_dim * axis_dim)
            * inner_dim + idx % inner_dim;
        for (int j = 0; j < axis_dim; j++)
            sum_val += x[offset + j * inner_dim];
        y[idx] = sum_val;
   }
}

template<> void Sum<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Sum<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 axis_dim, inner_dim, x, y);
}

template <typename T>
__global__ void _SumGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 coeff,
    const T*                dy,
    float*                  dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int offset = (idx / inner_dim * axis_dim)
            * inner_dim + idx % inner_dim;
        for (int j = 0; j < axis_dim; j++)
            dx[offset + j * inner_dim] = dy[idx] * coeff;
    }
}

template<> void SumGrad<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             coeff,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _SumGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 axis_dim, inner_dim, coeff, dy, dx);
}

/******************** ndarray.repeat ********************/

template <typename T>
__global__ void _Repeat(
    const int               count,
    const int               inner_dim,
    const int               repeats,
    const int               dim,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int d = idx % inner_dim;
        const int b = (idx / inner_dim / repeats) % dim;
        const int n = idx / inner_dim / repeats / dim;
        const int x_idx = (n * dim + b) * inner_dim + d;
        y[idx] = x[x_idx];
    }
}

template <> void Repeat<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Repeat<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 inner_dim, repeats, dim, x, y);
}

template <typename T>
__global__ void _RepeatGrad(
    const int               count,
    const int               inner_dim,
    const int               repeats,
    const int               dim,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int d = idx % inner_dim;
        const int b = (idx / inner_dim) % dim;
        const int n = idx / inner_dim  / dim;
        T gradient = 0;
        for (int t = 0; t < repeats; t++)
            gradient += dy[
                (((n * dim + b) * repeats) + t)
                    * inner_dim + d];
        dx[idx] = gradient;
    }
}

template <> void RepeatGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _RepeatGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, inner_dim, repeats, dim, dy, dx);
}

/******************** ndarray.slice ********************/

template <typename T>
__global__ void _Slice(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = y_slice_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int slice_idx = idx % tmp;
        const int x_idx = (outer_idx * x_slice_dim + slice_offset)
                                * inner_dim + slice_idx;
        y[idx] = x[x_idx];
    }
}

template <> void Slice<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Slice<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 outer_dim, inner_dim,
                    x_slice_dim, y_slice_dim,
                        slice_offset, x, y);
}

template <typename T>
__global__ void _SliceGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int tmp = y_slice_dim * inner_dim;
        const int outer_idx = idx / tmp;
        const int slice_idx = idx % tmp;
        const int x_idx = (outer_idx * x_slice_dim + slice_offset) 
                                * inner_dim + slice_idx;
        dx[x_idx] = dy[idx];
    }
}

template <> void SliceGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _SliceGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 outer_dim, inner_dim,
                     x_slice_dim, y_slice_dim,
                         slice_offset, dy,  dx);
}

/******************** ndarray.tile ********************/

template <typename T>
__global__ void _Tile(
    const int               count,
    const int               ex_inner_dim,
    const int               multiple,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int d = idx % ex_inner_dim;
        const int n = idx / ex_inner_dim / multiple;
        const int x_idx = n * ex_inner_dim + d;
        y[idx] = x[x_idx];
    }
}

template <> void Tile<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Tile<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 ex_inner_dim, multiple, x, y);
}

template <typename T>
__global__ void _TileGrad(
    const int               count,
    const int               ex_inner_dim,
    const int               multiple,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T gradient = 0;
        const int offset = (idx / ex_inner_dim * multiple)
            * ex_inner_dim + idx % ex_inner_dim;
        for (int t = 0; t < multiple; t++)
            gradient += dy[offset + t * ex_inner_dim];
        dx[idx] = gradient;
    }
}

template <> void TileGrad<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _TileGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
                 count, ex_inner_dim, multiple, dy, dx);
}

/******************** ndarray.transpose ********************/

template <typename T>
__global__ void _Transpose(
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

template <> void Transpose<float, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Transpose<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 ndim, order, old_steps, new_steps, x, y);
}

template <typename T>
__global__ void _TransposeGrad(
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

template <> void TransposeGrad<float, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _TransposeGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 ndim, order, old_steps, new_steps, dy, dx);
}

/******************** recurrent.lstm_cell ********************/

template <typename T>
__global__ void _LSTMCellAct(
    const int               count,
    const int               c_offset,
    const int               x_offset,
    T*                      xact) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int offset = idx % x_offset;
        xact[idx] = offset < c_offset ?
            _SigmoidUnit<float>(xact[idx]) : tanh(xact[idx]);
    }
}

template <typename T>
__global__ void _LSTMCellGate(
    const int               count,
    const int               hidden_size,
    const int               o_offset, // 2 * hidden_size
    const int               c_offset, // 3 * hidden_size
    const int               x_offset, // 4 * hidden_size
    const T*                cx,
    const T*                xact,
    T*                      c,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int n = idx / hidden_size;
        const int offset = idx % hidden_size;
        const T* x  = xact + n * x_offset;
        const T i = x[offset];
        const T f = x[offset + hidden_size];
        const T o = x[offset + o_offset];
        T c_ = x[offset + c_offset];
        c_ = c[idx] = f * cx[idx] + i * c_;
        h[idx] = o * tanh(c_);
    }
}

template <> void LSTMCell<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const float*            cx,
    float*                  xact,
    float*                  c,
    float*                  h,
    CUDAContext*            ctx) {
    const int o_offset = 2 * C,
                  c_offset = 3 * C,
                      x_offset = 4 * C;
    _LSTMCellAct<float>
        << < CUDA_BLOCKS(count * 4), CUDA_THREADS,
             0, ctx->cuda_stream() >> > (count * 4,
                 c_offset, x_offset, xact);
    _LSTMCellGate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 C, o_offset, c_offset, x_offset,
                    cx, xact, c, h);
}

template <typename T>
__global__ void _LSTMCellGateGrad(
    const int               count,
    const int               hidden_size,
    const int               o_offset,
    const int               c_offset,
    const int               x_offset,
    const T*                cx,
    const T*                xact,
    const T*                c,
    const T*                dc,
    const T*                dh,
    T*                      dcx,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int n = idx / hidden_size;
        const int offset = idx % hidden_size;
        const T* xact_ = xact + n * x_offset;
        T* dx_ = dx + n * x_offset;
        const T i = xact_[offset];
        const T f = xact_[offset + hidden_size];
        const T o = xact_[offset + o_offset];
        const T g = xact_[offset + c_offset];
        const T tanh_c = tanh(c[idx]);
        const T dcx_sum_term =
            dh[idx] * o * (1 - tanh_c * tanh_c) + dc[idx];
        dcx[idx] = dcx_sum_term * f;
        dx_[offset] = dcx_sum_term * g;
        dx_[offset + hidden_size] = dcx_sum_term * cx[idx];
        dx_[offset + o_offset] = dh[idx] * tanh_c;
        dx_[offset + c_offset] = dcx_sum_term * i;
    }
}

template <typename T>
__global__ void _LSTMCellActGrad(
    const int               count,
    const int               c_offset,
    const int               x_offset,
    const T*                xact,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int offset = idx % x_offset;
        const T val = xact[idx];
        if (offset < c_offset) dx[idx] = dx[idx] * val * (T(1) - val);
        else dx[idx] = dx[idx] * (T(1) - val * val);
    }
}

template <> void LSTMCellGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const float*            cx,
    const float*            xact,
    const float*            c,
    const float*            dc,
    const float*            dh,
    float*                  dcx,
    float*                  dx,
    CUDAContext*            ctx) {
    const int o_offset = 2 * C, 
                  c_offset = 3 * C,
                      x_offset = 4 * C;
    _LSTMCellGateGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 C, o_offset, c_offset, x_offset,
                    cx, xact, c, dc, dh, dcx, dx);
    _LSTMCellActGrad<float>
        << < CUDA_BLOCKS(count * 4), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count * 4,
                 c_offset, x_offset, xact, dx);
}

/******************** update.adam_update ********************/

template <typename T>
__global__ void _AdamUpdate(
    const int               count,
    const T                 lr,
    const T                 beta1,
    const T                 beta2,
    const T                 eps,
    T*                      g,
    T*                      m,
    T*                      v) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T gi = g[i];
        T mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
        T vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        g[i] = lr * mi / (sqrt(vi) + eps);
    }
}

template <> void AdamUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    float*                  g,
    float*                  m,
    float*                  v,
    CUDAContext*            ctx) {
    _AdamUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> > (count,
                 lr, beta1, beta2, eps, g, m, v);
}

/******************** update.nesterov_update ********************/

template <typename T>
__global__ void _NesterovUpdate(
    const int               count,
    const T                 lr,
    const T                 momentum,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T hi = h[i];
        T hi_new = h[i] = momentum * hi + lr * g[i];
        g[i] = (1 + momentum) * hi_new - momentum * hi;
    }
}

template <> void NesterovUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _NesterovUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> > (count,
                 lr, momentum, g, h);
}

/******************** update.rmsprop_update ********************/

template <typename T>
__global__ void _RMSPropUpdate(
    const int               count,
    const T                 lr,
    const T                 decay,
    const T                 eps,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T gi = g[i];
        T hi = h[i] = decay * h[i] + (1 - decay) * gi * gi;
        g[i] = lr * g[i] / (sqrt(hi) + eps);
    }
}

template <> void RMSPropUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _RMSPropUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 lr, decay, eps, g, h);
}

/******************** update.sgd_update ********************/

template <typename T>
__global__ void _SGDUpdate(
    const int               count,
    const T                 lr,
    const T                 momentum,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T hi = h[i];
        g[i] = h[i] = momentum * hi + lr * g[i];
    }
}

template <> void SGDUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _SGDUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 lr, momentum, g, h);
}

/******************** vision.bias_add ********************/

template <typename T>
__global__ void _BiasAdd_NCHW(
    const int               count,
    const int               dim,
    const int               inner_dim,
    const T*                bias,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] += bias[(idx / inner_dim) % dim];
    }
}

template <typename T>
__global__ void _BiasAdd_NHWC(
    const int               count,
    const int               dim,
    const int               inner_dim,
    const T*                bias,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] += bias[idx % dim];
    }
}

template<> void BiasAdd<float, CUDAContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const string&           data_format,
    const float*            bias,
    const float*            bias_multiplier,
    float*                  y,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _BiasAdd_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
                     count, dim, inner_dim, bias, y);
    } else if (data_format == "NHWC") {
        _BiasAdd_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
                     count, dim, inner_dim, bias, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.bilinear_resize ********************/

template <typename T>
__global__ void _BilinearResize_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int c = (idx / out_w / out_h) % C;
        const int n = idx / out_w / out_w / C;

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NCHT = (n * C + c) * H + top_y_idx;
        const int NCHB = (n * C + c) * H + bottom_y_idx;

        const float top_left(x[NCHT * W + left_x_idx]);
        const float top_right(x[NCHT * W + right_x_idx]);
        const float bottom_left(x[NCHB * W + left_x_idx]);
        const float bottom_right(x[NCHB * W + right_x_idx]);

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        y[idx] = top + (bottom - top) * y_lerp;
    }
}

template <typename T>
__global__ void _BilinearResize_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % out_w;
        const int h = (idx / C / out_w) % out_h;
        const int n = idx / C / out_w / out_h;

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NHT = n * H + top_y_idx;
        const int NHB = n * H + bottom_y_idx;

        const float top_left(x[(NHT * W + left_x_idx) * C + c]);
        const float top_right(x[(NHT * W + right_x_idx) * C + c]);
        const float bottom_left(x[(NHB * W + left_x_idx) * C + c]);
        const float bottom_right(x[(NHB * W + right_x_idx) * C + c]);

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        y[idx] = top + (bottom - top) * y_lerp;
    }
}

template <> void BilinearResize<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
     if (data_format == "NCHW") {
         _BilinearResize_NCHW<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(
                      count, N, C, H, W, out_h, out_w,
                          scale_h, scale_w, x, y);
    } else if(data_format == "NHWC") {
         _BilinearResize_NHWC<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(
                      count, N, C, H, W, out_h, out_w,
                          scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
__global__ void _BilinearResizeGrad_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int c = (idx / out_w / out_h) % C;
        const int n = idx / out_w / out_w / C;

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NCHT = (n * C + c) * H + top_y_idx;
        const int NCHB = (n * C + c) * H + bottom_y_idx;
        const float dtop = (1 - y_lerp) * dy[idx];
        const float dbottom = y_lerp * dy[idx];

        atomicAdd(&dx[NCHT * W + left_x_idx], static_cast<T>((1 - x_lerp) * dtop));
        atomicAdd(&dx[NCHT * W + right_x_idx], static_cast<T>(x_lerp * dtop));
        atomicAdd(&dx[NCHB * W + left_x_idx], static_cast<T>((1 - x_lerp) * dbottom));
        atomicAdd(&dx[NCHB * W + right_x_idx], static_cast<T>(x_lerp * dbottom));
    }
}

template <typename T>
__global__ void _BilinearResizeGrad_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % out_w;
        const int h = (idx / C / out_w) % out_h;
        const int n = idx / C / out_w / out_h;

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NHT = n * H + top_y_idx;
        const int NHB = n * H + bottom_y_idx;
        const float dtop = (1 - y_lerp) * dy[idx];
        const float dbottom = y_lerp * dy[idx];

        atomicAdd(&dx[(NHT * W + left_x_idx) * C + c], static_cast<T>((1 - x_lerp) * dtop));
        atomicAdd(&dx[(NHT * W + right_x_idx) * C + c], static_cast<T>(x_lerp * dtop));
        atomicAdd(&dx[(NHB * W + left_x_idx) * C + c], static_cast<T>((1 - x_lerp) * dbottom));
        atomicAdd(&dx[(NHB * W + right_x_idx) * C + c], static_cast<T>(x_lerp * dbottom));
    }
}

template <> void BilinearResizeGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
     if (data_format == "NCHW") {
         _BilinearResizeGrad_NCHW<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(
                      count, N, C, H, W, out_h, out_w,
                          scale_h, scale_w, dy, dx);
    } else if(data_format == "NHWC") {
         _BilinearResizeGrad_NHWC<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(
                      count, N, C, H, W, out_h, out_w,
                          scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.conv ********************/

template<typename T>
__global__ void _Im2Col2d_NCHW(
    const int               count,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % col_w;
        const int h_idx = idx / col_w;
        const int h = h_idx % col_h;
        const int im_c = h_idx / col_h;
        const int c = im_c * kernel_h * kernel_w;

        const int im_h_off = h * stride_h - pad_h;
        const int im_w_off = w * stride_w - pad_w;

        T* col_ptr = col;
        col_ptr += ((c * col_h + h) * col_w + w);

        const T* im_ptr = im;
        im_ptr += ((im_c * H + im_h_off) * W + im_w_off);

        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                const int im_h = kh * dilation_h + im_h_off;
                const int im_w = kw * dilation_w + im_w_off;
                *col_ptr = (im_h >= 0 && im_w >= 0 && im_h < H && im_w < W) ? 
                    im_ptr[kh * dilation_h * W + kw * dilation_w] : 0;
                col_ptr += (col_h * col_w);
            }
        }
    }
}

template<typename T>
__global__ void _Im2Col2d_NHWC(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % col_w;
        const int h = idx / C / col_w;
      
        const int im_h_off = h * stride_h - pad_h;
        const int im_w_off = w * stride_w - pad_w;
        const int base_col_idx = (h * col_w) + w;

        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                const int im_h = kh * dilation_h + im_h_off;
                const int im_w = kw * dilation_w + im_w_off;
                const int col_idx = (
                    ((base_col_idx * kernel_h + kh) * kernel_w + kw) * C + c
                );
                col[col_idx] = (im_h >= 0 && im_w >= 0 &&
                    im_h < H && im_w < W) ? im[(im_h * W + im_w) * C + c] : 0;
            }
        }
    }
}

template <> void Im2Col2d<float, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            im,
    float*                  col,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
         const int count = (C * col_h * col_w);
         _Im2Col2d_NCHW<float> 
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(count,
                      H, W, col_h, col_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w,
                              dilation_h, dilation_w, im, col);
    } else if (data_format == "NHWC") {
         const int count = (col_h * col_w * C);
         _Im2Col2d_NHWC<float> 
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(count, 
                      C, H, W, col_h, col_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w,
                              dilation_h, dilation_w, im, col);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
__global__ void _Col2Im2d_NCHW(
    const int               count,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T val = 0;
        const int im_w = idx % W + pad_w;
        const int im_h = (idx / W) % H + pad_h;
        const int im_c = idx / W / H;
        const int ex_kernel_h = (kernel_h - 1) * dilation_h + 1;
        const int ex_kernel_w = (kernel_w - 1) * dilation_w + 1;

        //  redundant pixels will be ignored when conv
        //  note to clip them by min(x,col_w)
        const int w_start = (im_w < ex_kernel_w) ?
            0 : (im_w - ex_kernel_w) / stride_w + 1;
        const int w_end = min(im_w / stride_w + 1, col_w);
        const int h_start = (im_h < ex_kernel_h) ? 
            0 : (im_h - ex_kernel_h) / stride_h + 1;
        const int h_end = min(im_h / stride_h + 1, col_h);

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int kh_off = (im_h - h * stride_h);
                int kw_off = (im_w - w * stride_w);
                //  only the serval im pixels used in dilated-conv
                //  ignore the corresponding col pixels
                if (kh_off % dilation_h == 0 && kw_off % dilation_w == 0) {
                    kh_off /= dilation_h;
                    kw_off /= dilation_w;
                    const int col_idx = ((
                        (im_c * kernel_h + kh_off) * kernel_w + kw_off) * col_h + h
                    ) * col_w + w;
                    val += col[col_idx];
                }
            }
        }
        im[idx] = val;
    }
}

template<typename T>
__global__ void _Col2Im2d_NHWC(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T val = 0;
        const int im_c = idx % C;
        const int im_w = (idx / C) % W + pad_w;
        const int im_h = (idx / C / W) + pad_h;
        const int ex_kernel_h = (kernel_h - 1) * dilation_h + 1;
        const int ex_kernel_w = (kernel_w - 1) * dilation_w + 1;

        //  redundant pixels will be ignored when conv
        //  note to clip them by min(x,col_w)
        const int w_start = (im_w < ex_kernel_w) ?
            0 : (im_w - ex_kernel_w) / stride_w + 1;
        const int w_end = min(im_w / stride_w + 1, col_w);
        const int h_start = (im_h < ex_kernel_h) ?
            0 : (im_h - ex_kernel_h) / stride_h + 1;
        const int h_end = min(im_h / stride_h + 1, col_h);

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int kh_off = (im_h - h * stride_h);
                int kw_off = (im_w - w * stride_w);
                //  only the serval im pixels used in dilated-conv
                //  ignore the corresponding col pixels
                if (kh_off % dilation_h == 0 && kw_off % dilation_w == 0) {
                    kh_off /= dilation_h;
                    kw_off /= dilation_w;
                    const int col_idx = (
                        ((h * col_w + w) * kernel_h + kh_off) * kernel_w + kw_off
                    ) * C + im_c;
                    val += col[col_idx];
                }
            }
        }
        im[idx] = val;
    }
}

template <> void Col2Im2d<float, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            col,
    float*                  im,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
         const int count = (C * H * W);
         _Col2Im2d_NCHW<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(count,
                      H, W, col_h, col_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w,
                              dilation_h, dilation_w, col, im);
    } else if (data_format == "NHWC") {
         const int count = (H * W * C);
         _Col2Im2d_NHWC<float> 
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >(count,
                      C, H, W, col_h, col_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w,
                              dilation_h, dilation_w, col, im);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.drop_block ********************/

template <typename T>
__global__ void _DropBlock2d_NCHW(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const uint32_t          thresh,
    const uint32_t*         seed,
    int*                    mask) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (seed[idx] < thresh) {
            const int x = idx % seed_w;
            const int y = (idx / seed_w) % seed_h;
            const int c = (idx / seed_w / seed_h) % C;
            const int n = (idx / seed_w / seed_h) / C;
            const int nc = (n * C + c) * H;
            for (int i = 0; i < block_size; ++i) {
                const int nch = (nc + y + i) * W;
                for (int j = 0; j < block_size; ++j)
                    atomicAnd(&mask[nch + x + j], 0);
            }
        }
    }
}

template <typename T>
__global__ void _DropBlock2d_NHWC(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const uint32_t          thresh,
    const uint32_t*         seed,
    int*                    mask) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        if (seed[idx] < thresh) {
            const int x = idx % seed_w;
            const int y = (idx / seed_w) % seed_h;
            const int c = (idx / seed_w / seed_h) % C;
            const int n = (idx / seed_w / seed_h) / C;
            for (int i = 0; i < block_size; ++i) {
                const int nh = (n * H + y + i) * W;
                for (int j = 0; j < block_size; ++j)
                    atomicAnd(&mask[(nh + x + j) * C + c], 0);
            }
        }
    }
}

template <> void DropBlock2d<CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const float             gamma,
    const string&           data_format,
    uint32_t*               seed,
    int*                    mask,
    CUDAContext*            ctx) {
    const int count = N * C * seed_h * seed_w;
    math::RandomUniform<uint32_t, CUDAContext>(
        count, 0.f, float(UINT_MAX), seed, ctx);
    auto thresh = static_cast<uint32_t>(UINT_MAX * gamma);
    if (data_format == "NCHW") {
        _DropBlock2d_NCHW<int>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     C, H, W, seed_h, seed_w, block_size,
                        thresh, seed, mask);
    } else if(data_format == "NHWC") {
        _DropBlock2d_NHWC<int>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                    C, H, W, seed_h, seed_w, block_size,
                        thresh, seed, mask);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.nn_resize ********************/

template <typename T>
__global__ void _NNResize_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int c = (idx / out_w / out_h) % C;
        const int n = idx / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        y[idx] = x[((n * C + c) * H + h_in) * W + w_in];
    }
}

template <typename T>
__global__ void _NNResize_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % out_w;
        const int h = (idx / C / out_w) % out_h;
        const int n = idx / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        y[idx] = x[((n * H + h_in) * W + w_in) * C + c];
    }
}

template <> void NNResize<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
                     count, N, C, H, W, out_h, out_w,
                         scale_h, scale_w, x, y);
    } else if(data_format == "NHWC") {
        _NNResize_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
                     count, N, C, H, W, out_h, out_w,
                         scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
 __global__ void _NNResizeGrad_NCHW(
     const int              count,
     const int              N,
     const int              C,
     const int              H,
     const int              W,
     const int              out_h,
     const int              out_w,
     const float            scale_h,
     const float            scale_w,
     const T*               dy,
     T*                     dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int c = (idx / out_w / out_h) % C;
        const int n = idx / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], dy[idx]);
    }
}

template <typename T>
__global__ void _NNResizeGrad_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % out_w;
        const int h = (idx / C / out_w) % out_h;
        const int n = idx / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], dy[idx]);
    }
}

template <> void NNResizeGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResizeGrad_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
                     count, N, C, H, W, out_h, out_w,
                         scale_h, scale_w, dy, dx);
    } else if(data_format == "NHWC") {
        _NNResizeGrad_NHWC<float> 
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
                     count, N, C, H, W, out_h, out_w,
                         scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.pooling ********************/

template<typename T>
__global__ void _MAXPooling2d_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pw = idx % pool_w;
        const int ph = (idx / pool_w) % pool_h;
        const int pc = (idx / pool_w / pool_h) % C;
        const int pn = idx / pool_w / pool_h / C;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        const int end_h = min(start_h + kernel_h, H);
        const int end_w = min(start_w + kernel_w, W);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);

        T max_val = -FLT_MAX;
        int max_idx = -1;
        const T* x_ptr = x + (pn * C + pc) * H * W;

        for (int h = start_h; h < end_h; ++h) {
            for (int w = start_w; w < end_w; ++w) {
                if (x_ptr[h * W + w] > max_val) {
                    max_idx = h * W + w;
                    max_val = x_ptr[max_idx];
                }
            }
        }
        y[idx] = max_val;
        mask[idx] = max_idx;
    }
}

template<typename T>
__global__ void _MAXPooling2d_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pc = idx % C;
        const int pw = (idx / C) % pool_w;
        const int ph = (idx / C / pool_w) % pool_h;
        const int pn = idx / C / pool_w / pool_h;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        const int end_h = min(start_h + kernel_h, H);
        const int end_w = min(start_w + kernel_w, W);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);

        T max_val = -FLT_MAX;
        int max_idx = -1;
        for (int h = start_h; h < end_h; ++h) {
            for (int w = start_w; w < end_w; ++w) {
                const int x_idx = ((pn * H + h) * W + w) * C + pc;
                if (x[x_idx] > max_val) {
                    max_idx = x_idx;
                    max_val = x[max_idx];
                }
            }
        }
        y[idx] = max_val;
        mask[idx] = max_idx;
    }
}

template<> void MAXPooling2d<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    int*                    mask,
    float*                  y,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _MAXPooling2d_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else if (data_format == "NHWC") {                        
        _MAXPooling2d_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
__global__ void _AVGPooling2d_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pw = idx % pool_w;
        const int ph = (idx / pool_w) % pool_h;
        const int pc = (idx / pool_w / pool_h) % C;
        const int pn = idx / pool_w / pool_h / C;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        int end_h = min(start_h + kernel_h, H + pad_h);
        int end_w = min(start_w + kernel_w, W + pad_w);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);
        end_h = min(end_h, H);
        end_w = min(end_w, W);

        const T* x_ptr = x + (pn * C + pc) * H * W;
        const int pool_area = (end_h - start_h) * (end_w - start_w);
        T avg_val = 0;

        for (int h = start_h; h < end_h; ++h) {
            for (int w = start_w; w < end_w; ++w) {
                avg_val += x_ptr[h * W + w];
            }
        }
        y[idx] = avg_val / pool_area;
    }
}

template<typename T>
__global__ void _AVGPooling2d_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pc = idx % C;
        const int pw = (idx / C) % pool_w;
        const int ph = (idx / C / pool_w) % pool_h;
        const int pn = idx / C / pool_w / pool_h;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        int end_h = min(start_h + kernel_h, H + pad_h);
        int end_w = min(start_w + kernel_w, W + pad_w);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);
        end_h = min(end_h, H);
        end_w = min(end_w, W);

        const int pool_area = (end_h - start_h) * (end_w - start_w);
        T avg_val = 0;

        for (int h = start_h; h < end_h; ++h) 
            for (int w = start_w; w < end_w; ++w)
                avg_val += x[((pn * H + h) * W + w) * C + pc];

        y[idx] = avg_val / pool_area;
    }
}

template<> void AVGPooling2d<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _AVGPooling2d_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, x, y);
    } else if (data_format == "NHWC") {
        _AVGPooling2d_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
__global__ void _MAXPooling2dGrad_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / W / H) % C;
        const int n = idx / W / H / C;

        //  allow overlapping
        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ? 
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        //  allow clip
        const int end_ph = min((h + pad_h) / stride_h + 1, pool_h);
        const int end_pw = min((w + pad_w) / stride_w + 1, pool_w);

        T grad = 0;
        const int offset = (n * C + c) * pool_h * pool_w;
        const T* dy_ptr = dy + offset;
        const int* mask_ptr = mask + offset;

        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                if (mask_ptr[ph * pool_w + pw] == (h * W + w)) {
                    grad += dy_ptr[ph * pool_w + pw];
                }
            }
        }
        dx[idx] = grad;
    }
}

template<typename T>
__global__ void _MAXPooling2dGrad_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % W;
        const int h = (idx / C / W) % H;
        const int n = idx / C / W / H;

        //  allow overlapping
        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        //  allow clip
        const int end_ph = min((h + pad_h) / stride_h + 1, pool_h);
        const int end_pw = min((w + pad_w) / stride_w + 1, pool_w);

        T grad = 0;
        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                const int x_idx = ((n * H + h) * W + w) * C + c;
                const int y_idx = ((n * pool_h + ph) * pool_w + pw) * C + c;
                if (mask[y_idx] == x_idx) grad += dy[y_idx];
            }
        }
        dx[idx] = grad;
    }
}

template<> void MAXPooling2dGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    const int*              mask,
    float*                  dx,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _MAXPooling2dGrad_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, dy,mask, dx);
    } else if (data_format == "NHWC") {
        _MAXPooling2dGrad_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, dy, mask, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
__global__ void _AVGPooling2dGrad_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / W / H) % C;
        const int n = idx / W / H / C;

        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int end_ph = min(h / stride_h + 1, pool_h);
        const int end_pw = min(w / stride_w + 1, pool_w);

        T grad = 0;
        const T* dy_ptr = dy + (n * C + c) * pool_h * pool_w;
        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = min(start_h + kernel_h, H + pad_h);
                int end_w = min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                grad += (dy_ptr[ph * pool_w + pw] / pool_area);
            }
        }
        dx[idx] = grad;
    }
}

template<typename T>
__global__ void _AVGPooling2dGrad_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % W;
        const int h = (idx / C / W) % H;
        const int n = idx / C / W / H;

        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int end_ph = min(h / stride_h + 1, pool_h);
        const int end_pw = min(w / stride_w + 1, pool_w);

        T grad = 0;
        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = min(start_h + kernel_h, H + pad_h);
                int end_w = min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                const int y_idx = ((n * pool_h + ph) * pool_w + pw) * C + c;
                grad += (dy[y_idx] / pool_area);
            }
        }
        dx[idx] = grad;
    }
}

template<> void AVGPooling2dGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
   if (data_format == "NCHW") {
        _AVGPooling2dGrad_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else if (data_format == "NHWC") {
        _AVGPooling2dGrad_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(count,
                     N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                         stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.roi_pooling ********************/

template <typename T>
__global__ void _ROIPooling(
    const int               count,
    const T                 spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const T*                x,
    const T*                rois,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            y[idx] = 0;
            mask[idx] = -1;
            continue;
        }

        int roi_start_w = round(offset_rois[1] * spatial_scale);
        int roi_start_h = round(offset_rois[2] * spatial_scale);
        int roi_end_w = round(offset_rois[3] * spatial_scale);
        int roi_end_h = round(offset_rois[4] * spatial_scale);

        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        const T bin_size_h = (T)roi_height / (T)pool_h;
        const T bin_size_w = (T)roi_width / (T)pool_w;

        int hstart = floor(bin_size_h * ph);
        int wstart = floor(bin_size_w * pw);
        int hend = ceil(bin_size_h * (ph + 1));
        int wend = ceil(bin_size_w * (pw + 1));

        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);

        bool is_empty = (hend <= hstart) || (wend <= wstart);
        float max_val = is_empty ? 0 : -FLT_MAX;
        int max_idx = -1;
        x += ((roi_batch_ind * channels + c) * height * width);
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                const int x_idx = h * width + w;
                if (x[x_idx] > max_val) {
                    max_val = x[x_idx];
                    max_idx = x_idx;
                }
            }
        }
        y[idx] = max_val;
        mask[idx] = max_idx;
    }
}

template<> void ROIPooling<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            x,
    const float*            rois,
    int*                    mask,
    float*                  y,
    CUDAContext*            ctx) {
    _ROIPooling<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 spatial_scale, C, H, W,
                     pool_h, pool_w, x, rois, mask, y);
}

template <typename T>
__global__ void _ROIPoolingGrad(
    const int               count,
    const int               num_rois,
    const T                 spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const T*                dy,
    const T*                rois,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / width / height) % channels;
        int n = idx / width / height / channels;

        T gradient = 0;

        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            const T* offset_rois = rois + roi_n * 5;
            int roi_batch_ind = offset_rois[0];

            if (n != roi_batch_ind) continue;

            int roi_start_w = round(offset_rois[1] * spatial_scale);
            int roi_start_h = round(offset_rois[2] * spatial_scale);
            int roi_end_w = round(offset_rois[3] * spatial_scale);
            int roi_end_h = round(offset_rois[4] * spatial_scale);

            const bool in_roi = (w >= roi_start_w &&
                                 w <= roi_end_w &&
                                 h >= roi_start_h &&
                                 h <= roi_end_h);

            if (!in_roi) continue;

            int y_offset = (roi_n * channels + c) * pool_h * pool_w;
            const T* offset_dy = dy + y_offset;
            const int* offset_mask = mask + y_offset;

            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            const T bin_size_h = (T)roi_height / (T)pool_h;
            const T bin_size_w = (T)roi_width / (T)pool_w;

            int phstart = floor(static_cast<T>(h - roi_start_h) / bin_size_h);
            int phend = ceil(static_cast<T>(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor(static_cast<T>(w - roi_start_w) / bin_size_w);
            int pwend = ceil(static_cast<T>(w - roi_start_w + 1) / bin_size_w);

            phstart = min(max(phstart, 0), pool_h);
            phend = min(max(phend, 0), pool_h);
            pwstart = min(max(pwstart, 0), pool_w);
            pwend = min(max(pwend, 0), pool_w);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int pool_idx = ph * pool_w + pw;
                    if (offset_mask[pool_idx] == (h * width + w)) {
                        gradient += offset_dy[pool_idx];
                    }
                }
            }
        }
        dx[idx] = gradient;
    }
}

template<> void ROIPoolingGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            dy,
    const float*            rois,
    const int*              mask,
    float*                  dx,
    CUDAContext*            ctx) {
    _ROIPoolingGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 num_rois, spatial_scale, C, H, W,
                     pool_h, pool_w, dy, rois, mask, dx);
}

/******************** vision.roi_align ********************/

template <typename T>
__device__ T _ROIAlignInterpolate(
    const T*                Xdata,
    const int               height,
    const int               width,
    T                       y,
    T                       x) {
    if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    T v1 = Xdata[y_low * width + x_low];
    T v2 = Xdata[y_low * width + x_high];
    T v3 = Xdata[y_high * width + x_low];
    T v4 = Xdata[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <typename T>
__global__ void _ROIAlign(
    const int               count,
    const float             spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const int               sampling_ratio,
    const T*                Xdata,
    const T*                rois,
    T*                      Ydata) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            Ydata[idx] = 0;
            continue;
        }

        T roi_start_w = offset_rois[1] * spatial_scale;
        T roi_start_h = offset_rois[2] * spatial_scale;
        T roi_end_w = offset_rois[3] * spatial_scale;
        T roi_end_h = offset_rois[4] * spatial_scale;

        T roi_width = max(roi_end_w - roi_start_w, (T)1.);
        T roi_height = max(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pool_h);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pool_w);

        const T* offset_Xdata = Xdata +(roi_batch_ind * channels + c) * height * width;

        int roi_bin_grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_height / pool_h);
        int roi_bin_grid_w = (sampling_ratio > 0) ? 
            sampling_ratio : ceil(roi_width / pool_w);

        const T num_bin_grids = roi_bin_grid_h * roi_bin_grid_w;

        T output_val = 0.;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = roi_start_w + pw * bin_size_w + 
                    static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
                T val = _ROIAlignInterpolate(offset_Xdata, height, width, y, x);
                output_val += val;
            }
        }
        output_val /= num_bin_grids;
        Ydata[idx] = output_val;
    }
}

template<> void ROIAlign<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            x,
    const float*            rois,
    float*                  y,
    CUDAContext*            ctx) {
    _ROIAlign<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 spatial_scale, C, H, W, pool_h, pool_w,
                     sampling_ratio, x, rois, y);
}

template <typename T>
__device__ void _ROIAlignInterpolateGrad(
    const int               height,
    const int               width,
    T                       y,
    T                       x,
    T&                      w1,
    T&                      w2,
    T&                      w3,
    T&                      w4,
    int&                    x_low,
    int&                    x_high,
    int&                    y_low,
    int&                    y_high) {
    if (y < -1.0 || y > height ||
            x < -1.0 || x > width) {
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    y_low = (int)y;
    x_low = (int)x;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    return;
}

template <typename T>
__global__ void _ROIAlignGrad(
    const int               count,
    const int               num_rois,
    const T                 spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const int               sampling_ratio,
    const T*                dYdata,
    const T*                rois,
    T*                      dXdata) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) continue;

        T roi_start_w = offset_rois[1] * spatial_scale;
        T roi_start_h = offset_rois[2] * spatial_scale;
        T roi_end_w = offset_rois[3] * spatial_scale;
        T roi_end_h = offset_rois[4] * spatial_scale;

        T roi_width = max(roi_end_w - roi_start_w, (T)1.);
        T roi_height = max(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pool_h);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pool_w);

        T* offset_dXdata = dXdata + 
            (roi_batch_ind * channels + c) * height * width;

        int y_offset = (n * channels + c) * pool_h * pool_w;
        const T* offset_dYdata = dYdata + y_offset;
        const T dYdata_this_bin = offset_dYdata[ph * pool_w + pw];

        int roi_bin_grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_height / pool_h);
        int roi_bin_grid_w = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_width / pool_w);

        const T num_bin_grids = roi_bin_grid_h * roi_bin_grid_w;

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

                T w1, w2, w3, w4;
                int x_low, x_high, y_low, y_high;

                _ROIAlignInterpolateGrad(
                    height, width, y, x, w1, w2, w3, w4,
                        x_low, x_high, y_low, y_high);

                T g1 = dYdata_this_bin * w1 / num_bin_grids;
                T g2 = dYdata_this_bin * w2 / num_bin_grids;
                T g3 = dYdata_this_bin * w3 / num_bin_grids;
                T g4 = dYdata_this_bin * w4 / num_bin_grids;

                if (x_low >= 0 && x_high >= 0 
                        && y_low >= 0 && y_high >= 0) {
                    atomicAdd(
                        offset_dXdata + y_low * width + x_low,
                        static_cast<T>(g1));
                    atomicAdd(
                        offset_dXdata + y_low * width + x_high,
                        static_cast<T>(g2));
                    atomicAdd(
                        offset_dXdata + y_high * width + x_low,
                        static_cast<T>(g3));
                    atomicAdd(
                        offset_dXdata + y_high * width + x_high,
                        static_cast<T>(g4));
                }
            }
        }
    }
}

template<> void ROIAlignGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            dy,
    const float*            rois,
    float*                  dx,
    CUDAContext*            ctx) {
    _ROIAlignGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(count,
                 num_rois, spatial_scale, C, H, W,
                     pool_h, pool_w, sampling_ratio, dy, rois, dx);
}

}    // namespace kernel

}    // namespace dragon

#endif // WITH_CUDA