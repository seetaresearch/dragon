#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _AffineNoBias(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const T*                x,
    const T*                alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int pi = (i / inner_dim) % axis_dim;
#if __CUDA_ARCH__ >= 350
        y[i] = __ldg(alpha + pi) * x[i];
#else
        y[i] = alpha[pi] * x[i];
#endif
    }
}

template <typename T>
__global__ void _Affine(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const T*                x,
    const T*                alpha,
    const T*                beta,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int pi = (i / inner_dim) % axis_dim;
#if __CUDA_ARCH__ >= 350
        y[i] = __ldg(alpha + pi) * x[i]
             + __ldg(beta + pi);
#else
        y[i] = alpha[pi] * x[i] + beta[pi];
#endif
    }
}

template<> void Affine<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            x,
    const float*            alpha,
    const float*            beta,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * axis_dim * inner_dim;
    if (beta != nullptr) {
        _Affine
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, axis_dim, inner_dim,
            x, alpha, beta, y
        );
    } else {
        _AffineNoBias
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, axis_dim, inner_dim, x, alpha, y
        );
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _AffineNoBias<half>(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const half*             x,
    const half*             alpha,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int pi = (i / inner_dim) % axis_dim;
        const float X32 = __half2float(x[i]);
        const float A32 = __half2float(__ldg(alpha + pi));
        y[i] = __float2half(A32 * X32);
#endif
    }
}

template<> __global__ void _Affine<half>(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const half*             x,
    const half*             alpha,
    const half*             beta,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int pi = (i / inner_dim) % axis_dim;
        const float X32 = __half2float(x[i]);
        const float A32 = __half2float(__ldg(alpha + pi));
        const float B32 = __half2float(__ldg(beta + pi));
        y[i] = __float2half(A32 * X32 + B32);
#endif
    }
}

template<> void Affine<float16, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          x,
    const float16*          alpha,
    const float16*          beta,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * axis_dim * inner_dim;
    if (beta != nullptr) {
        _Affine
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, axis_dim, inner_dim,
            reinterpret_cast<const half*>(x),
            reinterpret_cast<const half*>(alpha),
            reinterpret_cast<const half*>(beta),
            reinterpret_cast<half*>(y)
        );
    } else {
        _AffineNoBias
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, axis_dim, inner_dim,
            reinterpret_cast<const half*>(x),
            reinterpret_cast<const half*>(alpha),
            reinterpret_cast<half*>(y)
         );
    }
}

/* <T = float32, Device = CUDA> */

template <> void AffineGrad<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            dy,
    const float*            alpha,
    float*                  dx,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * axis_dim * inner_dim;
    _AffineNoBias
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads, axis_dim, inner_dim, dy, alpha, dx
    );
}

/* <T = float16, Device = CUDA> */

template <> void AffineGrad<float16, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float16*          dy,
    const float16*          alpha,
    float16*                dx,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * axis_dim * inner_dim;
    _AffineNoBias
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads, axis_dim, inner_dim,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(alpha),
        reinterpret_cast<half*>(dx)
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA