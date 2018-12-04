#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Affine <T = float32, Device = CUDA> */

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
                 0, ctx->cuda_stream() >> >
            (count, scale_dim, inner_dim, x, alpha, beta, y);
    } else {
        _AffineWithOBias<float>
            << <CUDA_BLOCKS(count), CUDA_THREADS,
                0, ctx->cuda_stream() >> >
            (count, scale_dim, inner_dim, x, alpha, y);
    }
}

/*! Affine <T = float16, Device = CUDA> */

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
        // ComputeType: float32
        const float x_fp32 = __half2float(x[idx]);
        const float alpha_fp32 = __half2float(alpha[scale_idx]);
        y[idx] = __float2half(alpha_fp32 * x_fp32);
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
        // ComputeType: float32
        const float x_fp32 = __half2float(x[idx]);
        const float alpha_fp32 = __half2float(alpha[scale_idx]);
        const float beta_fp32 = __half2float(beta[scale_idx]);
        y[idx] = __float2half(alpha_fp32 * x_fp32 + beta_fp32);
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
                 0, ctx->cuda_stream() >> >
            (count, scale_dim, inner_dim,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<const half*>(alpha),
                        reinterpret_cast<const half*>(beta),
                            reinterpret_cast<half*>(y));
    } else {
        _AffineWithOBiasHalf<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, scale_dim, inner_dim,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<const half*>(alpha),
                        reinterpret_cast<half*>(y));
    }
}

/*! AffineGrad <T = float32, Device = CUDA> */

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
             0, ctx->cuda_stream() >> >
        (count, scale_dim, inner_dim, dy, alpha, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA