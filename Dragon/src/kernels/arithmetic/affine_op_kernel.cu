#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Affine <T = float32, Device = CUDA> */

template <typename T>
__global__ void _AffineNoBias(
    const int               nthreads,
    const int               inner_dim,
    const int               scale_dim,
    const T*                x,
    const T*                alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
        const int scale_idx = (idx / inner_dim) % scale_dim;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(alpha + scale_idx) * x[idx];
#else
        y[idx] = alpha[scale_idx] * x[idx];
#endif
    }
}

template <typename T>
__global__ void _Affine(
    const int               nthreads,
    const int               inner_dim,
    const int               scale_dim,
    const T*                x,
    const T*                alpha,
    const T*                beta,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
        const int scale_idx = (idx / inner_dim) % scale_dim;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(alpha + scale_idx) * x[idx] +
                      __ldg(beta + scale_idx);
#else
        y[idx] = alpha[scale_idx] * x[idx] + beta[scale_idx];
#endif
    }
}

template<> void Affine<float, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float*            x,
    const float*            alpha,
    const float*            beta,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * scale_dim * inner_dim;
    if (beta != nullptr) {
        _Affine<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, inner_dim, scale_dim, x, alpha, beta, y);
    } else {
        _AffineNoBias<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, inner_dim, scale_dim, x, alpha, y);
    }
}

/*! Affine <T = float16, Device = CUDA> */

__global__ void _AffineNoBiasHalf(
    const int               nthreads,
    const int               inner_dim,
    const int               scale_dim,
    const half*             x,
    const half*             alpha,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int scale_idx = (idx / inner_dim) % scale_dim;
        const float X32 = __half2float(x[idx]);
        const float A32 = __half2float(__ldg(alpha + scale_idx));
        y[idx] = __float2half(A32 * X32);
#endif
    }
}

__global__ void _AffineHalf(
    const int               nthreads,
    const int               inner_dim,
    const int               scale_dim,
    const half*             x,
    const half*             alpha,
    const half*             beta,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int scale_idx = (idx / inner_dim) % scale_dim;
        const float X32 = __half2float(x[idx]);
        const float A32 = __half2float(__ldg(alpha + scale_idx));
        const float B32 = __half2float(__ldg(beta + scale_idx));
        y[idx] = __float2half(A32 * X32 + B32);
#endif
    }
}

template<> void Affine<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float16*          x,
    const float16*          alpha,
    const float16*          beta,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * scale_dim * inner_dim;
    if (beta != nullptr) {
        _AffineHalf
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, inner_dim, scale_dim,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<const half*>(alpha),
                        reinterpret_cast<const half*>(beta),
                            reinterpret_cast<half*>(y));
    } else {
        _AffineNoBiasHalf
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, inner_dim, scale_dim,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<const half*>(alpha),
                        reinterpret_cast<half*>(y));
    }
}

/*! AffineGrad <T = float32, Device = CUDA> */

template <> void AffineGrad<float, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float*            dy,
    const float*            alpha,
    float*                  dx,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * scale_dim * inner_dim;
    _AffineNoBias<float>
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, inner_dim, scale_dim, dy, alpha, dx);
}

/*! AffineGrad <T = float16, Device = CUDA> */

template <> void AffineGrad<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float16*          dy,
    const float16*          alpha,
    float16*                dx,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * scale_dim * inner_dim;
    _AffineNoBiasHalf
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, inner_dim, scale_dim,
            reinterpret_cast<const half*>(dy),
                reinterpret_cast<const half*>(alpha),
                    reinterpret_cast<half*>(dx));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA