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
__global__ void _SetHalf(
    const int               n,
    const T                 alpha,
    T*                      x) {
    CUDA_KERNEL_LOOP(idx, n) {
        x[idx] = alpha;
    }
}

template <> void Set<float16, CUDAContext>(
    const int               n,
    const float16           alpha,
    float16*                x) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _SetHalf<half2>
            << <CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                dragon_cast<half2, float16>(alpha),
                    reinterpret_cast<half2*>(x));
    } else {
        _SetHalf<float16>
            << <CUDA_BLOCKS(n), CUDA_THREADS >> >(n, alpha, x);
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
__global__ void _TypeFloat2Half(
    const int               n,
    const float*            a,
    half*                   b) {
    CUDA_KERNEL_LOOP(idx, n) {
        b[idx] = __float2half(a[idx]);
    }
}
#endif

template <> void RandomNormal<float16, CUDAContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    float16*                x,
    CUDAContext*            ctx) {
#ifdef WITH_CUDA_FP16
    float* xf32 = (float*)CUDAContext::New(n * sizeof(float));
    CURAND_CHECK(curandGenerateNormal(
        ctx->curand_generator(), xf32, n, mu, sigma));
    _TypeFloat2Half
        << <CUDA_BLOCKS(n), CUDA_THREADS >> >(
            n, xf32, reinterpret_cast<half*>(x));
    CUDAContext::Delete(xf32);
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

/******************** Level-1 ********************/

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _AddHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hadd(a[idx], b[idx]);
#endif
    }
}

template <typename T>
__global__ void _AddHalf2(
    const int               n,
    const half2*            a,
    const half2*            b,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hadd2(a[idx], b[idx]);
#endif
    }
}
#endif

template <> void Add<float16, CUDAContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _AddHalf2<half2>
            << <CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                reinterpret_cast<const half2*>(a),
                    reinterpret_cast<const half2*>(b),
                        reinterpret_cast<half2*>(y));
    } else {
        _AddHalf<half>
            << <CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
                reinterpret_cast<const half*>(a),
                    reinterpret_cast<const half*>(b),
                        reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _SubHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hsub(a[idx], b[idx]);
#endif
    }
}

template <typename T>
__global__ void _SubHalf2(
    const int               n,
    const half2*            a,
    const half2*            b,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hsub2(a[idx], b[idx]);
#endif
    }
}
#endif

template <> void Sub<float16, CUDAContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _SubHalf2<half2>
            << <CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                reinterpret_cast<const half2*>(a),
                    reinterpret_cast<const half2*>(b),
                        reinterpret_cast<half2*>(y));
    } else {
        _SubHalf<half>
            << <CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
                reinterpret_cast<const half*>(a),
                    reinterpret_cast<const half*>(b),
                        reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _MulHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(a[idx], b[idx]);
#endif
    }
}

template <typename T>
__global__ void _MulHalf2(
    const int               n,
    const half2*            a,
    const half2*            b,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(a[idx], b[idx]);
#endif
    }
}
#endif

template <> void Mul<float16, CUDAContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _MulHalf2<half2>
            << <CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                reinterpret_cast<const half2*>(a),
                    reinterpret_cast<const half2*>(b),
                        reinterpret_cast<half2*>(y));
    } else {
        _MulHalf<half>
            << <CUDA_BLOCKS(n), CUDA_THREADS >> > (n,
                reinterpret_cast<const half*>(a),
                    reinterpret_cast<const half*>(b),
                        reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _DivHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hdiv(a[idx], b[idx]);
#endif
    }
}
#endif

template <> void Div<float16, CUDAContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    _DivHalf<half>
        << <CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
            reinterpret_cast<const half*>(a),
                reinterpret_cast<const half*>(b),
                    reinterpret_cast<half*>(y));
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _SquareHalf(
    const int               n,
    const half*             x,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(x[idx], x[idx]);
#endif
    }
}

template <typename T>
__global__ void _SquareHalf2(
    const int               n,
    const half2*            x,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(x[idx], x[idx]);
#endif
    }
}
#endif

template <> void Square<float16, CUDAContext>(
    int                     n,
    const float16*          x,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _SquareHalf2<half2>
            << < CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                reinterpret_cast<const half2*>(x),
                    reinterpret_cast<half2*>(y));
    } else {
        _SquareHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS >> > (n,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _SqrtHalf(
    int                     n,
    const half*             x,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = hsqrt(x[idx]);
#endif
    }
}

template <typename T>
__global__ void _SqrtHalf2(
    const int               n,
    const half2*            x,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = h2sqrt(x[idx]);
#endif
    }
}
#endif

template <> void Sqrt<float16, CUDAContext>(
    int                     n,
    const float16*          x,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _SqrtHalf2<half2>
            << < CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                reinterpret_cast<const half2*>(x),
                    reinterpret_cast<half2*>(y));
    } else {
        _SqrtHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _PowHalf(
    const int               n,
    const float             alpha,
    const half*             a,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(a[idx], a[idx]);
#endif
    }
}

template <typename T>
__global__ void _PowHalf2(
    const int               n,
    const float             alpha,
    const half2*            a,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(a[idx], a[idx]);
#endif
    }
}
#endif

template <> void Pow<float16, CUDAContext>(
    int                     n,
    const float             alpha,
    const float16*          x,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    CHECK(alpha == float(2)) << "fp16 only support the power of 2";
    if (n % 2 == 0) {
        _PowHalf2<half2>
            << < CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                alpha, reinterpret_cast<const half2*>(x),
                    reinterpret_cast<half2*>(y));
    } else {
        _PowHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
                alpha, reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _InvHalf(
    const int               n,
    const half              numerator,
    const half*             x,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] =  __hmul(hrcp(x[idx]), numerator);
#endif
    }
}

template <typename T>
__global__ void _InvHalf2(
    const int               n,
    const half2             numerator,
    const half2*            x,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(h2rcp(x[idx]), numerator);
#endif
    }
}
#endif

template <> void Inv<float16, CUDAContext>(
    const int               n,
    const float             numerator,
    const float16*          x,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _InvHalf2<half2>
            << < CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                dragon_cast<half2, float>(numerator),
                    reinterpret_cast<const half2*>(x),
                        reinterpret_cast<half2*>(y));
    } else {
        _InvHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
                dragon_cast<half, float>(numerator),
                    reinterpret_cast<const half*>(x),
                        reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

/******************** Level-2 ********************/

template <> void Scal<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    float16*                y,
    CUDAContext*            ctx) {
#ifdef WITH_CUDA_FP16
    CUBLAS_CHECK(cublasScalEx(
        ctx->cublas_handle(), n,
            &alpha, CUDA_R_32F,
                y, CUDA_R_16F, 1,
                    CUDA_R_32F));
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

template <> void Scale<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    CUDAContext::Copy<float16, CUDAContext, CUDAContext>(n, y, x);
    Scal<float16, CUDAContext>(n, alpha, y, ctx);
}

template <> float Dot<float16, CUDAContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    CUDAContext*            ctx) {
#ifdef WITH_CUDA_FP16
    float16 result;
    CUBLAS_CHECK(cublasDotEx(
        ctx->cublas_handle(), n,
            a, CUDA_R_16F, 1,
                b, CUDA_R_16F, 1,
                    &result, CUDA_R_16F,
                        CUDA_R_32F));
    return dragon_cast<float, float16>(result);
#else
    CUDA_FP16_NOT_COMPILED;
    return 0.;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _AddScalarHalf(
    const int               n,
    half                    alpha,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hadd(y[idx], alpha);
#endif
    }
}

template <typename T>
__global__ void _AddScalarHalf2(
    const int               n,
    half2                   alpha,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hadd2(y[idx], alpha);
#endif
    }
}
#endif

template <> void AddScalar<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _AddScalarHalf2<half2>
            << <CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                dragon_cast<half2, float>(alpha),
                    reinterpret_cast<half2*>(y));
    } else {
        _AddScalarHalf<half>
            << <CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
                dragon_cast<half, float>(alpha),
                    reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

#ifdef WITH_CUDA_FP16
template <typename T>
__global__ void _MulScalarHalf(
    const int               n,
    half                    alpha,
    half*                   y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(y[idx], alpha);
#endif
    }
}

template <typename T>
__global__ void _MulScalarHalf2(
    const int               n,
    half2                   alpha,
    half2*                  y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(y[idx], alpha);
#endif
    }
}
#endif

template <> void MulScalar<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    float16*                y) {
#ifdef WITH_CUDA_FP16
    if (n % 2 == 0) {
        _MulScalarHalf2<half2>
            << <CUDA_BLOCKS(n / 2), CUDA_THREADS >> >(n / 2,
                dragon_cast<half2, float>(alpha),
                    reinterpret_cast<half2*>(y));
    } else {
        _MulScalarHalf<half>
            << <CUDA_BLOCKS(n), CUDA_THREADS >> >(n,
                dragon_cast<half, float>(alpha),
                    reinterpret_cast<half*>(y));
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

template <> void Axpy<float16, CUDAContext>(
    const int               n,
    float                   alpha,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
#ifdef WITH_CUDA_FP16
    CUBLAS_CHECK(cublasAxpyEx(
        ctx->cublas_handle(), n,
            &alpha, CUDA_R_32F,
                x, CUDA_R_16F, 1,
                    y, CUDA_R_16F, 1,
                        CUDA_R_32F));
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

template <> void Axpby<float16, CUDAContext>(
    const int               n,
    float                   alpha,
    const float16*          x,
    float                   beta,
    float16*                y,
    CUDAContext*            ctx) {
    Scal<float16, CUDAContext>(n, beta, y, ctx);
    Axpy<float16, CUDAContext>(n, alpha, x, y, ctx);
}

template <> void RandomUniform<float16, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    float16*                x,
    CUDAContext*            ctx) {
#ifdef WITH_CUDA_FP16
    float* xf32 = (float*)ctx->New(n * sizeof(float));
    CURAND_CHECK(curandGenerateUniform(
        ctx->curand_generator(), xf32, n));
    _TypeFloat2Half
        << <CUDA_BLOCKS(n), CUDA_THREADS >> >(
            n, xf32, reinterpret_cast<half*>(x));
    float range = high - low;
    if (range != float(1)) Scal<float16, CUDAContext>(n, range, x, ctx);
    if (low != float(0)) AddScalar<float16, CUDAContext>(n, low, x);
    ctx->Delete(xf32);
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

/******************** Level-3 ********************/

template <> void Gemm<float16, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const CBLAS_TRANSPOSE   TransB,
    const int               M,
    const int               N,
    const int               K,
    const float             alpha,
    const float16*          A,
    const float16*          B,
    const float             beta,
    float16*                C,
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
#ifdef WITH_CUDA_FP16
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (TransB == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    if (math_type == TensorProto_DataType_FLOAT) {
        const float _alpha_ = alpha, _beta_ = beta;
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMM + MATH32 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    B, CUDA_R_16F, ldb,
                    A, CUDA_R_16F, lda,
                &_beta_,
                    C, CUDA_R_16F, N,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMM + MATH32 + DEFAULT
            CUBLAS_CHECK(cublasSgemmEx(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    B, CUDA_R_16F, ldb,
                    A, CUDA_R_16F, lda,
                &_beta_,
                    C, CUDA_R_16F, N));
        }
#else
       CUBLAS_CHECK(cublasSgemmEx(
           ctx->cublas_handle(),
           cuTransB, cuTransA, N, M, K,
           &_alpha_,
               B, CUDA_R_16F, ldb,
               A, CUDA_R_16F, lda,
           &_beta_,
               C, CUDA_R_16F, N));
#endif
    } else if (math_type == TensorProto_DataType_FLOAT16) {
        const half _alpha_ = dragon_cast<half, float>(alpha);
        const half _beta_ = dragon_cast<half, float>(beta);
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMM + MATH16 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    B, CUDA_R_16F, ldb,
                    A, CUDA_R_16F, lda,
                &_beta_,
                    C, CUDA_R_16F, N,
                CUDA_R_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMM + MATH16 + DEFAULT
            CUBLAS_CHECK(cublasHgemm(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    reinterpret_cast<const half*>(B), ldb,
                    reinterpret_cast<const half*>(A), lda,
                &_beta_,
                    reinterpret_cast<half*>(C), N));
        }
#else
        CUBLAS_CHECK(cublasHgemm(
            ctx->cublas_handle(),
            cuTransB, cuTransA, N, M, K,
            &_alpha_,
                reinterpret_cast<const half*>(B), ldb,
                reinterpret_cast<const half*>(A), lda,
            &_beta_,
                reinterpret_cast<half*>(C), N));
#endif
    } else {
        LOG(FATAL) << "Unsupported math type";
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

template <> void Gemv<float16, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const float16*          A,
    const float16*          x,
    const float             beta,
    float16*                y,
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
#ifdef WITH_CUDA_FP16
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_T : CUBLAS_OP_N;
    int m = (cuTransA == CUBLAS_OP_N) ? N : M;
    int k = (cuTransA == CUBLAS_OP_N) ? M : N;
    int LDA = (cuTransA == CUBLAS_OP_N) ? m : k;
    int LDC = m;
    const float _alpha_ = alpha, _beta_ = beta;
    if (math_type == TensorProto_DataType_FLOAT) {
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMV + MATH32 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    A, CUDA_R_16F, LDA,
                    x, CUDA_R_16F, k,
                &_beta_,
                    y, CUDA_R_16F, LDC,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMV + MATH32 + DEFAULT
            CUBLAS_CHECK(cublasSgemmEx(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    A, CUDA_R_16F, LDA,
                    x, CUDA_R_16F, k,
                &_beta_,
                    y, CUDA_R_16F, LDC));
        }
#else
        CUBLAS_CHECK(cublasSgemmEx(
            ctx->cublas_handle(),
            cuTransA, CUBLAS_OP_N, m, 1, k,
            &_alpha_,
                A, CUDA_R_16F, LDA,
                x, CUDA_R_16F, k,
            &_beta_,
                y, CUDA_R_16F, LDC));
#endif
    } else if (math_type == TensorProto_DataType_FLOAT16) {
        const half _alpha_ = dragon_cast<half, float>(alpha);
        const half _beta_ = dragon_cast<half, float>(beta);
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMV + MATH16 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    A, CUDA_R_16F, LDA,
                    x, CUDA_R_16F, k,
                &_beta_,
                    y, CUDA_R_16F, LDC,
                CUDA_R_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMV + MATH16 + DEFAULT
            CUBLAS_CHECK(cublasHgemm(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    reinterpret_cast<const half*>(A), LDA,
                    reinterpret_cast<const half*>(x), k,
                &_beta_,
                    reinterpret_cast<half*>(y), LDC));
        }
#else
        CUBLAS_CHECK(cublasHgemm(
            ctx->cublas_handle(),
            cuTransA, CUBLAS_OP_N, m, 1, k,
            &_alpha_,
                reinterpret_cast<const half*>(A), LDA,
                reinterpret_cast<const half*>(x), k,
            &_beta_,
                reinterpret_cast<half*>(y), LDC));
#endif
    } else {
            LOG(FATAL) << "Unsupported math type";
    }
#else
    CUDA_FP16_NOT_COMPILED;
#endif
}

}    // namespace math

}    // namespace dragon

#endif // WITH_CUDA