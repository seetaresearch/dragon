#ifdef WITH_CUDA

#include <cmath>

#include "core/context_cuda.h"
#include "utils/cuda_device.h"
#include "utils/math_functions.h"
#include "utils/cast.h"

namespace dragon {

namespace math {

/******************** Level-0 ********************/

template <typename T>
__global__ void _Set(
    const int               n,
    const T                 alpha,
    T*                      x) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        x[idx] = alpha;
    }
}

template <> void Set<float, CUDAContext>(
    const int               n,
    const float             alpha,
    float*                  x,
    CUDAContext*            ctx) {
    if (alpha == 0.f) {
        CUDA_CHECK(cudaMemsetAsync(x, 0,
            sizeof(float) * n, ctx->cuda_stream()));
    } else {
        _Set<float>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n, alpha, x);
    }
}

template <> void Set<int, CUDAContext>(
    const int               n,
    const int               alpha,
    int*                    x,
    CUDAContext*            ctx) {
    if (alpha == 0) {
        CUDA_CHECK(cudaMemsetAsync(x, 0,
            sizeof(int) * n, ctx->cuda_stream()));
    } else {
        _Set<int>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n, alpha, x);
    }
}

template <> void Set<int64_t, CUDAContext>(
    const int               n,
    const int64_t           alpha,
    int64_t*                x,
    CUDAContext*            ctx) {
    if (alpha == 0) {
        CUDA_CHECK(cudaMemsetAsync(x, 0,
            sizeof(int64_t) * n, ctx->cuda_stream()));
    }
    else {
        _Set<int64_t>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n, alpha, x);
    }
}

template <> void RandomUniform<uint32_t, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    uint32_t*               x,
    CUDAContext*            ctx) {
    //  note that we ignore the low / high
    //  curand could only generates in the range of [0, uint32]
    auto* rng = ctx->curand_generator();
    CURAND_CHECK(curandGenerate(rng, x, n));
}

template <> void RandomNormal<float, CUDAContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    float*                  x,
    CUDAContext*            ctx) {
    auto* rng = ctx->curand_generator();
    CURAND_CHECK(curandGenerateNormal(rng, x, n, mu, sigma));
}

/******************** Level-1 ********************/

template <typename T>
__global__ void _Add(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] + b[idx];
    }
}

template <> void Add<float, CUDAContext>(
    int                     n,
    const float*            a,
    const float*            b,
    float*                  y,
    CUDAContext*            ctx) {
    _Add<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, a, b, y);
}

template <typename T>
__global__ void _Sub(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] - b[idx];
    }
}

template <> void Sub<float, CUDAContext>(
    int                     n,
    const float*            a,
    const float*            b,
    float*                  y,
    CUDAContext*            ctx) {
    _Sub<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, a, b, y);
}

template <typename T>
__global__ void _Mul(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] * b[idx];
    }
}

template <> void Mul<float, CUDAContext>(
    int                     n,
    const float*            a,
    const float*            b,
    float*                  y,
    CUDAContext*            ctx) {
    _Mul<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, a, b, y);
}

template <typename T>
__global__ void _Div(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] / b[idx];
    }
}

template <> void Div<float, CUDAContext>(
    int                     n,
    const float*            a,
    const float*            b,
    float*                  y,
    CUDAContext*            ctx) {
    _Div<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, a, b, y);
}

template <typename T>
__global__ void _Clip(
    const int               n,
    const T                 low,
    const T                 high,
    T*                      x) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        x[idx] = x[idx] > high ? high : x[idx];
        x[idx] = x[idx] < low ? low : x[idx];
    }
}

template <> void Clip<float, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    float*                  x,
    CUDAContext*            ctx) {
    _Clip<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, low, high, x);
}

template <typename T>
__global__ void _Exp(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = exp(a[idx]);
    }
}

template <> void Exp<float, CUDAContext>(
    int                     n,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Exp<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, x, y);
}

template <typename T>
__global__ void _Log(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = log(a[idx]);
    }
}

template <> void Log<float, CUDAContext>(
    int                     n,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Log<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, x, y);
}

template <typename T>
__global__ void _Square(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = x[idx] * x[idx];
    }
}

template <> void Square<float, CUDAContext>(
    int                     n,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Square<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, x, y);
}

template <typename T>
__global__ void _Sqrt(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = sqrt(x[idx]);
    }
}

template <> void Sqrt<float, CUDAContext>(
    int                     n,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Sqrt<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, x, y);
}

template <typename T>
__global__ void _Pow(
    const int               n,
    const T                 alpha,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = pow(a[idx], alpha);
    }
}

template <> void Pow<float, CUDAContext>(
    int                     n,
    const float             alpha,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Pow<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, alpha, x, y);
}

template <typename T>
__global__ void _Inv(
    const int               n,
    const float             numerator,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] = numerator / x[idx];
    }
}

template <> void Inv<float, CUDAContext>(
    const int               n,
    const float             numerator,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Inv<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, numerator, x, y);
}

/******************** Level-2 ********************/

template <> void Scal<float, CUDAContext>(
    const int               n,
    const float             alpha,
    float*                  y,
    CUDAContext*            ctx) {
    CUBLAS_CHECK(cublasSscal_v2(
        ctx->cublas_handle(), n, &alpha, y, 1));
}

template <> void Scale<float, CUDAContext>(
    const int               n,
    const float             alpha,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    CUBLAS_CHECK(cublasScopy_v2(
        ctx->cublas_handle(), n, x, 1, y, 1));
    CUBLAS_CHECK(cublasSscal_v2(
        ctx->cublas_handle(), n, &alpha, y, 1));
}

template <> void StridedDot<float, CUDAContext>(
    const int               n,
    const float*            a,
    const int               incx,
    const float*            b,
    const int               incy,
    float*                  y,
    CUDAContext*            ctx) {
    CUBLAS_CHECK(cublasSdot_v2(ctx->cublas_handle(),
        n, a, incx, b, incy, y));
}

template <> void Dot<float, CUDAContext>(
    int                     n,
    const float*            a,
    const float*            b,
    float*                  y,
    CUDAContext*            ctx) {
    StridedDot<float, CUDAContext>(
        n, a, 1, b, 1, y, ctx);
    ctx->FinishDeviceCompution();
}

template <> float ASum<float, CUDAContext>(
    const int               n,
    const float*            x) {
    return cublasSasum(n, x, 1);
}

template <typename T>
__global__ void _AddScalar(
    const int               n,
    T                       alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] += alpha;
    }
}

template <> void AddScalar<float, CUDAContext>(
    const int               n,
    const float             alpha,
    float*                  y,
    CUDAContext*            ctx) {
    _AddScalar<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, alpha, y);
}

template <typename T>
__global__ void _MulScalar(
    const int               n,
    T                       alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, n) {
        y[idx] *= alpha;
    }
}

template <> void MulScalar<float, CUDAContext>(
    const int               n,
    const float             alpha,
    float*                  y,
    CUDAContext*            ctx) {
    _MulScalar<float>
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(n, alpha, y);
}

template <> void Axpy<float, CUDAContext>(
    const int               n,
    float                   alpha,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    CUBLAS_CHECK(cublasSaxpy_v2(
        ctx->cublas_handle(), n, &alpha, x, 1, y, 1));
}

template <> void Axpby<float, CUDAContext>(
    const int               n,
    float                   alpha,
    const float*            x,
    float                   beta,
    float*                  y,
    CUDAContext*            ctx) {
    Scal<float, CUDAContext>(n, beta, y, ctx);
    Axpy<float, CUDAContext>(n, alpha, x, y, ctx);
}

template <> void RandomUniform<float, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    float*                  x,
    CUDAContext*            ctx) {
    CURAND_CHECK(curandGenerateUniform(
        ctx->curand_generator(), x, n));
    float range = high - low;
    if (range != 1.f) Scal<float, CUDAContext>(n, range, x, ctx);
    if (low != 0.f) AddScalar<float, CUDAContext>(n, low, x, ctx);
}

/******************** Level-3 ********************/

template <> void Gemm<float, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const CBLAS_TRANSPOSE   TransB,
    const int               M,
    const int               N,
    const int               K,
    const float             alpha,
    const float*            A,
    const float*            B,
    const float             beta,
    float*                  C,
    CUDAContext*            ctx,
    TensorProto_DataType math_type) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (TransB == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    const float _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasSgemm_v2(ctx->cublas_handle(),
        cuTransB, cuTransA, N, M, K,
            &_alpha_, B, ldb, A, lda, &_beta_, C, N));
}

template <> void Gemv<float, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const float*            A,
    const float*            x,
    const float             beta,
    float*                  y,
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_T : CUBLAS_OP_N;
    const float _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasSgemv_v2(
        ctx->cublas_handle(), cuTransA, N, M,
            &_alpha_, A, N, x, 1, &_beta_, y, 1));
}

}    // namespace math

}    // namespace dragon

#endif // WITH_CUDA