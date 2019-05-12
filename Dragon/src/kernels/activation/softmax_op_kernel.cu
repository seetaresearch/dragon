#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SoftmaxReduceMax(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const T*                x,
    T*                      scale) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const T* X = x + (
            (i / inner_dim) *
                axis_dim * inner_dim
        ) + (i % inner_dim);
        T val = *X;
        for (int c = 1; c < axis_dim; c++)
            val = max(X[c * inner_dim], val);
        scale[i] = val;
    }
}

template <typename T>
__global__ void _SoftmaxSub(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const T*                scale,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int iix = i % inner_dim;
        const int oix = i / inner_dim / axis_dim;
#if __CUDA_ARCH__ >= 350
        y[i] -= __ldg(scale + (oix * inner_dim + iix));
#else
        y[i] -= scale[oix * inner_dim + iix];
#endif
    }
}

template <typename T>
__global__ void _SoftmaxReduceSum(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const T*                y,
    T*                      scale) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const T* Y = y + (
            (i / inner_dim) *
                axis_dim * inner_dim
        ) + (i % inner_dim);
        T val = *Y;
        for (int c = 1; c < axis_dim; c++)
            val += Y[c * inner_dim];
        scale[i] = val;
    }
}

template <typename T>
 __global__ void _SoftmaxDiv(
     const int              nthreads,
     const int              axis_dim,
     const int              inner_dim,
     const T*               scale,
     T*                     y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int iix = i % inner_dim;
        const int oix = i / inner_dim / axis_dim;
#if __CUDA_ARCH__ >= 350
        y[i] /= __ldg(scale + (oix * inner_dim + iix));
#else
        y[i] /= scale[oix * inner_dim + iix];
#endif
    }
}

template<> void Softmax<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            multiplier,
    const float*            x,
    float*                  scale,
    float*                  y,
    CUDAContext*            ctx) {
    auto num_preds = outer_dim * inner_dim;
    auto nelements = num_preds * axis_dim;
    _SoftmaxReduceMax
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        num_preds, axis_dim, inner_dim, x, scale
    );
    _SoftmaxSub
        << < CUDA_BLOCKS(nelements), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nelements, axis_dim, inner_dim, scale, y
    );

    math::Exp(nelements, y, y, ctx);

    _SoftmaxReduceSum
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        num_preds, axis_dim, inner_dim, y, scale
    );
    _SoftmaxDiv
        << < CUDA_BLOCKS(nelements), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nelements, axis_dim, inner_dim, scale, y
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SoftmaxDot(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const T*                dy,
    const T*                y,
    T*                      scale) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const T* dY = dy + (
            (i / inner_dim) *
                axis_dim * inner_dim
        ) + (i % inner_dim);
        const T* Y = y + (
            (i / inner_dim) *
                axis_dim * inner_dim
        ) + (i % inner_dim);
        T val = (*dY) * (*Y);
        for (int c = 1; c < axis_dim; c++)
            val += dY[c * inner_dim] * Y[c * inner_dim];
        scale[i] = val;
    }
}

template<> void SoftmaxGrad<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            multiplier,
    const float*            dy,
    const float*            y,
    float*                  scale,
    float*                  dx,
    CUDAContext*            ctx) {
    auto num_preds = outer_dim * inner_dim;
    auto nelements = num_preds * axis_dim;
    _SoftmaxDot
        << < CUDA_BLOCKS(num_preds), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        num_preds, axis_dim, inner_dim, dy, y, scale
    );
    _SoftmaxSub
        << < CUDA_BLOCKS(nelements), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        nelements, axis_dim, inner_dim, scale, dx
    );
    math::Mul(nelements, dx, y, dx, ctx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA