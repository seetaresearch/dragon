#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/*! Softmax <T = float32, Device = CUDA> */

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
    T*                      scale) {
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
             0, ctx->cuda_stream() >> >
        (outer_dim, classes, inner_dim, x, scale);

    _SoftmaxSubtract<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, classes, inner_dim, scale, y);

    _SoftmaxExp<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, y);

    _SoftmaxSumClass<float>
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (outer_dim, classes, inner_dim, y, scale);

    _SoftmaxDiv<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, classes, inner_dim, scale, y);
}

/*! SoftmaxGrad <T = float32, Device = CUDA> */

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
             0, ctx->cuda_stream() >> >
        (outer_dim, classes, inner_dim, dy, y, scale);

    _SoftmaxSubtract<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, classes,inner_dim, scale, dx);

    math::Mul<float, CUDAContext>(count, dx, y, dx, ctx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA