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
__global__ void _Set(const int n, const T alpha, T* x) {
    CUDA_KERNEL_LOOP(idx, n) {
        x[idx] = alpha;
    }
}

template <> void Set<float, CUDAContext>(const int n, 
                                         const float alpha, 
                                         float* x) {
    if (alpha == 0) {
        CUDA_CHECK(cudaMemset(x, 0, sizeof(float) * n));
        return;
    }
    _Set<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, alpha, x);
}

template <> void Set<int, CUDAContext>(const int n, 
                                       const int alpha, 
                                       int* x) {
    if (alpha == 0) {
        CUDA_CHECK(cudaMemset(x, 0, sizeof(int) * n));
        return;
    }
    _Set<int> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, alpha, x);
}

template <typename T>
__global__ void _SetHalf2(const int n, const half2 alpha, half2* x) {
    CUDA_KERNEL_LOOP(idx, n) {
        x[idx] = alpha;
    }
}

template <> void Set<float16, CUDAContext>(const int n, 
                                           const float16 alpha, 
                                           float16* x) {
    if (n % 2 == 0) {
        float16 alpha_fp16 = alpha;
        half2 alpha_fp32;
        alpha_fp32.x = dragon_cast<float32, float16>(alpha_fp16).x;
        _SetHalf2<half2> << <GET_BLOCKS(n / 2), CUDA_NUM_THREADS >> >(n / 2, 
                                                                 alpha_fp32, 
                                               reinterpret_cast<half2*>(x));
    } else {
        _Set<float16> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, alpha, x);
    }
}

template <> void RandomUniform<uint32_t, CUDAContext>(const int n, 
                                                      const float low, 
                                                      const float high, 
                                                      uint32_t* x) {
    //  note that we ignore the low / high
    //  curand could only generates in the range of [0, uint32]
    CURAND_CHECK(curandGenerate(curand_generator(), x, n));
}

template <> void RandomUniform<float16, CUDAContext>(const int n,
                                                     const float low,
                                                     const float high,
                                                     float16* x) {
    NOT_IMPLEMENTED;
}

template <> void RandomNormal<float, CUDAContext>(const int n,
                                                  const float mu, 
                                                  const float sigma, 
                                                  float* x) {
    CURAND_CHECK(curandGenerateNormal(curand_generator(), x, n, mu, sigma));
}

template <> void RandomNormal<float16, CUDAContext>(const int n,
                                                    const float mu, 
                                                    const float sigma, 
                                                    float16* x) {
    NOT_IMPLEMENTED;
}

template <> void RandomBernoulli<float, CUDAContext>(const int n,
                                                     const float p, 
                                                     unsigned int* x) {
    //  curand could not generate bernoulli distribution
    //  we recommend implement it within specfic case, e.g. Dropout
    NOT_IMPLEMENTED;
}

/******************** Level-1 ********************/

template <typename T>
__global__ void _Add(const int n, const T* a, const T *b, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] + b[idx];
    }
}

template <> void Add<float, CUDAContext>(int n, 
                                         const float* a, 
                                         const float* b, 
                                         float *y) {
    _Add<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, a, b, y);
}

template <typename T>
__global__ void _Sub(const int n, const T* a, const T *b, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] - b[idx];
    }
}

template <> void Sub<float, CUDAContext>(int n, 
                                         const float* a, 
                                         const float* b, 
                                         float *y) {
    _Sub<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, a, b, y);
}

template <typename T>
__global__ void _Mul(const int n, const T* a, const T* b, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] * b[idx];
    }
}
    
template <> void Mul<float, CUDAContext>(int n, 
                                         const float* a, 
                                         const float* b, 
                                         float* y) {
    _Mul<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, a, b, y);
}

template <typename T>
__global__ void _MulHalf(const int n, const half* a, const half* b, half* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(a[idx], b[idx]);
#endif
    }
}

template <typename T>
__global__ void _MulHalf2(const int n, const half2* a, const half2* b, half2* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(a[idx], b[idx]);
#endif
    }
}
    
template <> void Mul<float16, CUDAContext>(int n, 
                                           const float16* a, 
                                           const float16* b, 
                                           float16* y) {
    if (n % 2 == 0)
        _MulHalf2<half2> << <GET_BLOCKS(n / 2), CUDA_NUM_THREADS >> >(n / 2, 
                                  reinterpret_cast<const half2*>(a),
                                  reinterpret_cast<const half2*>(b),
                                  reinterpret_cast<half2*>(y));
    else _MulHalf<half> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, 
                                  reinterpret_cast<const half*>(a),
                                  reinterpret_cast<const half*>(b),
                                  reinterpret_cast<half*>(y));
}

template <typename T>
__global__ void _Div(const int n, const T* a, const T* b, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] / b[idx];
    }
}

template <> void Div<float, CUDAContext>(int n, 
                                         const float* a, 
                                         const float* b, 
                                         float* y) {
    _Div<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, a, b, y);
}

template <typename T>
__global__ void _DivHalf(const int n, const half* a, const half* b, half* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = hdiv(a[idx], b[idx]);
#endif
    }
}

template <> void Div<float16, CUDAContext>(int n, 
                                           const float16* a, 
                                           const float16* b,
                                           float16* y) {
    _DivHalf<half> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, 
                             reinterpret_cast<const half*>(a),
                             reinterpret_cast<const half*>(b),
                             reinterpret_cast<half*>(y));
}

template <typename T>
__global__ void _Clip(const int n, const T low, const T high, T* x) {
    CUDA_KERNEL_LOOP(idx, n) {
        x[idx] = x[idx] > high ? high : x[idx];
        x[idx] = x[idx] < low ? low : x[idx];
    }
}

template <> void Clip<float, CUDAContext>(const int n,
                                          const float low,
                                          const float high,
                                          float* x) {
    _Clip<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, low, high, x);
}

template <typename T>
__global__ void _Exp(const int n, const T* a, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = std::exp(a[idx]);
    }
}

template <> void Exp<float, CUDAContext>(int n, const float* x, float *y) {
    _Exp<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, x, y);
}

template <typename T>
__global__ void _Log(const int n, const T* a, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = std::log(a[idx]);
    }
}

template <> void Log<float, CUDAContext>(int n, const float* x, float *y) {
    _Log<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, x, y);
}

template <typename T>
__global__ void _Square(const int n, const T* x, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = x[idx] * x[idx];
    }
}

template <> void Square<float, CUDAContext>(int n, 
                                            const float* x, 
                                            float* y) {
    _Square<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, x, y);
}

template <typename T>
__global__ void _SquareHalf(const int n, const half* x, half* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(x[idx], x[idx]);
#endif
    }
}

template <typename T>
__global__ void _SquareHalf2(const int n, const half2* x, half2* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(x[idx], x[idx]);
#endif
    }
}

template <> void Square<float16, CUDAContext>(int n,
                                              const float16* x,
                                              float16* y) {
    if (n % 2 == 0)
        _SquareHalf2<half2> << < GET_BLOCKS(n / 2), CUDA_NUM_THREADS >> >(n / 2,
                                              reinterpret_cast<const half2*>(x),
                                              reinterpret_cast<half2*>(y));
    else _SquareHalf<half> << < GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n,
                                      reinterpret_cast<const half*>(x),
                                      reinterpret_cast<half*>(y));
    CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void _Sqrt(const int n, const T* x, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = std::sqrt(x[idx]);
    }
}

template <> void Sqrt<float, CUDAContext>(int n, 
                                          const float* x, 
                                          float* y) {
    _Sqrt<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, x, y);
}

template <typename T>
__global__ void _SqrtHalf(const int n, const half* x, half* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = hsqrt(x[idx]);
#endif
    }
}

template <typename T>
__global__ void _SqrtHalf2(const int n, const half2* x, half2* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = h2sqrt(x[idx]);
#endif
    }
}

template <> void Sqrt<float16, CUDAContext>(int n,
                                            const float16* x,
                                            float16* y) {
    if (n % 2 == 0)
        _SqrtHalf2<half2> << < GET_BLOCKS(n / 2), CUDA_NUM_THREADS >> >(n / 2,
                                            reinterpret_cast<const half2*>(x),
                                            reinterpret_cast<half2*>(y));
    else _SqrtHalf<half>  << < GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n,
                                     reinterpret_cast<const half*>(x),
                                     reinterpret_cast<half*>(y));
    CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void _Pow(const int n, const T alpha, const T* a, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = std::pow(a[idx], alpha);
    }
}

template <> void Pow<float, CUDAContext>(int n, 
                                         const float alpha, 
                                         const float* x, 
                                         float* y) {
    _Pow<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, alpha, x, y);
}

template <typename T>
__global__ void _PowHalf(const int n, const float alpha, const half* a, half* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(a[idx], a[idx]);
#endif
    }
}

template <typename T>
__global__ void _PowHalf2(const int n, const float alpha, const half2* a, half2* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(a[idx], a[idx]);
#endif
    }
}

template <> void Pow<float16, CUDAContext>(int n, 
                                           const float alpha, 
                                           const float16* x, 
                                           float16* y) {
    CHECK(alpha == float(2)) << "fp16 only support the power of 2";
    if (n % 2 == 0)
        _PowHalf2<half2> << < GET_BLOCKS(n / 2), CUDA_NUM_THREADS >> >(n / 2,
                                                                       alpha,
                                            reinterpret_cast<const half2*>(x),
                                            reinterpret_cast<half2*>(y));

    else _PowHalf<half>  << < GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, 
                                                               alpha, 
                                    reinterpret_cast<const half*>(x), 
                                    reinterpret_cast<half*>(y));
    CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void _Inv(const int n, const float numerator, const T* x, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = numerator / x[idx];
    }
}

template <> void Inv<float, CUDAContext>(const int n, 
                                         const float numerator, 
                                         const float* x, 
                                         float* y) {
    _Inv<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, numerator, x, y);
}

template <typename T>
__global__ void _InvHalf(const int n, const half numerator, const half* x, half* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] =  __hmul(hrcp(x[idx]), numerator);
#endif
    }
}

template <typename T>
__global__ void _InvHalf2(const int n, const half2 numerator, const half2* x, half2* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul2(h2rcp(x[idx]), numerator);
#endif
    }
}

template <> void Inv<float16, CUDAContext>(const int n, 
                                           const float numerator, 
                                           const float16* x, 
                                           float16* y) {
    half numerator_fp16;
    numerator_fp16.x = dragon_cast<float16, float>(numerator).x;
    if (n % 2 == 0) {
        half2 numerator_fp32;
        numerator_fp32.x = dragon_cast<float32, float>(numerator).x;
        _InvHalf2<half2> << < GET_BLOCKS(n / 2), CUDA_NUM_THREADS >> >(n / 2,
                                                               numerator_fp32,
                                            reinterpret_cast<const half2*>(x),
                                                reinterpret_cast<half2*>(y));
    }
    else {
        _InvHalf<half> << < GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n,
                                                    numerator_fp16,
                                  reinterpret_cast<const half*>(x),
                                       reinterpret_cast<half*>(y));
    }
    CUDA_POST_KERNEL_CHECK;
}

/******************** Level-2 ********************/

template <> void Scal<float, CUDAContext>(const int n, const float alpha, float* y) {
    CUBLAS_CHECK(cublasSscal_v2(cublas_handle(), 
                                n, 
                                &alpha, 
                                y, 1));
}

template <> void Scal<float16, CUDAContext>(const int n, const float alpha, float16* y) {
    CUBLAS_CHECK(cublasScalEx(cublas_handle(), 
                              n, 
                              &alpha, CUDA_R_32F, 
                              y, CUDA_R_16F, 1, 
                              CUDA_R_32F));
}

template <> void Scale<float, CUDAContext>(const int n, 
                                           const float alpha, 
                                           const float* x, 
                                           float* y) {
    CUBLAS_CHECK(cublasScopy_v2(cublas_handle(), n, x, 1, y, 1));
    CUBLAS_CHECK(cublasSscal_v2(cublas_handle(), n, &alpha, y, 1));
}

template <> void Scale<float16, CUDAContext>(const int n, 
                                             const float alpha, 
                                             const float16* x, 
                                             float16* y) {
    CUDAContext ctx;
    ctx.Copy<float16, CUDAContext, CUDAContext>(n, y, x);
    Scal<float16, CUDAContext>(n, alpha, y);
}

template <> float StridedDot<float, CUDAContext>(const int n,
                                                 const float* a, 
                                                 const int incx, 
                                                 const float* b, 
                                                 const int incy) {
    float result;
    CUBLAS_CHECK(cublasSdot_v2(cublas_handle(), 
                               n, 
                               a, incx, 
                               b, incy, 
                               &result));
    return result;
}

template <> float Dot<float, CUDAContext>(int n, const float* a, const float* b) {
    return StridedDot<float, CUDAContext>(n, a, 1, b, 1);
}

template <> float Dot<float16, CUDAContext>(int n, const float16* a, const float16* b) {
    float16 result;
    CUBLAS_CHECK(cublasDotEx(cublas_handle(),
                             n,
                             &a, CUDA_R_16F, 1,
                             &b, CUDA_R_16F, 1,
                             &result, CUDA_R_16F,
                             CUDA_R_32F));
    return dragon_cast<float, float16>(result);
}

template <> float ASum<float, CUDAContext>(const int n, const float* x) {
    return cublasSasum(n, x, 1);
}

template <typename T>
__global__ void _AddScalar(const int n, T alpha, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] += alpha;
    }
}

template <> void AddScalar<float, CUDAContext>(const int n, const float alpha, float* y) {
    _AddScalar<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, alpha, y);
}

template <typename T>
__global__ void _AddScalarHalf(const int n, half alpha, half* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hadd(y[idx], alpha);
#endif
    }
}

template <typename T>
__global__ void _AddScalarHalf2(const int n, half2 alpha, half2* y) {
    CUDA_KERNEL_LOOP(idx, n) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hadd2(y[idx], alpha);
#endif
    }
}

template <> void AddScalar<float16, CUDAContext>(const int n, const float alpha, float16* y) {
    half alpha_fp16;
    alpha_fp16.x = dragon_cast<float16, float>(alpha).x;
    if (n % 2 == 0) {
        half2 alpha_fp32;
        alpha_fp32.x = dragon_cast<float32, float>(alpha).x;
        _AddScalarHalf2<half2> << <GET_BLOCKS(n / 2), CUDA_NUM_THREADS >> >(n / 2,
                                                                       alpha_fp32,
                                                     reinterpret_cast<half2*>(y));
    } else {
        _AddScalarHalf<half> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n,
                                                             alpha_fp16,
                                            reinterpret_cast<half*>(y));
    }
    CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void _MulScalar(const int n, T alpha, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] *= alpha;
    }
}

template <> void MulScalar<float, CUDAContext>(const int n, const float alpha, float* y) {
    _MulScalar<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, alpha, y);
}

template <> void Axpy<float, CUDAContext>(const int n, 
                                          float alpha, 
                                          const float* x, 
                                          float* y) {
    CUBLAS_CHECK(cublasSaxpy_v2(cublas_handle(), 
                                n, 
                                &alpha, x, 1, 
                                y, 1));
}

template <> void Axpy<float16, CUDAContext>(const int n, 
                                            float alpha, 
                                            const float16* x, 
                                            float16* y) {
    CUBLAS_CHECK(cublasAxpyEx(cublas_handle(), 
                              n, 
                              &alpha, CUDA_R_32F, 
                              x, CUDA_R_16F, 1, 
                              y, CUDA_R_16F, 1, 
                              CUDA_R_32F));
}

template <> void Axpby<float, CUDAContext>(const int n, 
                                           float alpha, 
                                           const float* x, 
                                           float beta, 
                                           float* y) {
    Scal<float, CUDAContext>(n, beta, y);
    Axpy<float, CUDAContext>(n, alpha, x, y);
}

template <> void Axpby<float16, CUDAContext>(const int n, 
                                             float alpha, 
                                             const float16* x, 
                                             float beta, 
                                             float16* y) {
    Scal<float16, CUDAContext>(n, beta, y);
    Axpy<float16, CUDAContext>(n, alpha, x, y);
}

/******************** Level-3 ********************/

template <> void RandomUniform<float, CUDAContext>(const int n, 
                                                   const float low, 
                                                   const float high, 
                                                   float* x) {
    CURAND_CHECK(curandGenerateUniform(curand_generator(), x, n));
    float range = high - low;
    if (range != float(1)) Scal<float, CUDAContext>(n, range, x);
    if (low != float(0)) AddScalar<float, CUDAContext>(n, low, x);
}

template <> void Gemm<float, CUDAContext>(const CBLAS_TRANSPOSE transA, 
                                          const CBLAS_TRANSPOSE transB,
                                          const int M, 
                                          const int N, 
                                          const int K, 
                                          const float alpha, 
                                          const float* A,
                                          const float* B, 
                                          const float beta, 
                                          float *C, 
                                          TensorProto_DataType math_type) {
    int lda = (transA == CblasNoTrans) ? K : M;
    int ldb = (transB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm_v2(cublas_handle(), 
                                cuTransB, cuTransA,
                                N, M, K, 
                                &alpha, 
                                B, ldb, 
                                A, lda, 
                                &beta, 
                                C, N));
}

template <> void Gemm<float16, CUDAContext>(const CBLAS_TRANSPOSE transA, 
                                            const CBLAS_TRANSPOSE transB,
                                            const int M, 
                                            const int N, 
                                            const int K, 
                                            const float alpha, 
                                            const float16* A,
                                            const float16* B, 
                                            const float beta, 
                                            float16 *C, 
                                            TensorProto_DataType math_type) {
    int lda = (transA == CblasNoTrans) ? K : M;
    int ldb = (transB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (math_type == TensorProto_DataType_FLOAT) {
        CUBLAS_CHECK(cublasSgemmEx(cublas_handle(), 
                                   cuTransB, cuTransA,
                                   N, M, K, 
                                   &alpha, 
                                   B, CUDA_R_16F, ldb, 
                                   A, CUDA_R_16F, lda,
                                   &beta, 
                                   C, CUDA_R_16F, N));
    } else if (math_type == TensorProto_DataType_FLOAT16) {
        half alpha_fp16;
        alpha_fp16.x = dragon_cast<float16, float>(alpha).x;
        half beta_fp16;
        beta_fp16.x = dragon_cast<float16, float>(beta).x;
        CUBLAS_CHECK(cublasHgemm(cublas_handle(), 
                                 cuTransB, cuTransA,
                                 N, M, K, 
                                 &alpha_fp16, 
                                 reinterpret_cast<const half*>(B), ldb,
                                 reinterpret_cast<const half*>(A), lda,
                                 &beta_fp16, 
                                 reinterpret_cast<half*>(C), N));
    } else {
        LOG(FATAL) << "unsupported math type";
    }
}

template <> void Gemv<float, CUDAContext>(const CBLAS_TRANSPOSE transA, 
                                          const int M, const int N,
                                          const float alpha, 
                                          const float* A, 
                                          const float* x, 
                                          const float beta, 
                                          float* y, 
                                          TensorProto_DataType math_type) { 
    cublasOperation_t cuTransA = (transA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(cublasSgemv_v2(cublas_handle(), 
                                cuTransA, 
                                N, M, 
                                &alpha, 
                                A, N, 
                                x, 1, 
                                &beta, 
                                y, 1));
}

template <> void Gemv<float16, CUDAContext>(const CBLAS_TRANSPOSE transA, 
                                            const int M, 
                                            const int N,
                                            const float alpha, 
                                            const float16* A, 
                                            const float16* x, 
                                            const float beta, 
                                            float16* y, 
                                            TensorProto_DataType math_type) {
    cublasOperation_t cuTransA = (transA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    int m = (cuTransA == CUBLAS_OP_N) ? N : M;
    int k = (cuTransA == CUBLAS_OP_N) ? M : N;
    int LDA = (cuTransA == CUBLAS_OP_N) ? m : k;
    int LDC = m;
    if (math_type == TensorProto_DataType_FLOAT) {
        CUBLAS_CHECK(cublasSgemmEx(cublas_handle(), 
                                   cuTransA, CUBLAS_OP_N,
                                    m, 1, k, 
                                    &alpha, 
                                    A, CUDA_R_16F, LDA, 
                                    x, CUDA_R_16F, k, 
                                    &beta,
                                    y, CUDA_R_16F, LDC));
    } else if (math_type == TensorProto_DataType_FLOAT16) {
        half alpha_fp16;
        alpha_fp16.x = dragon_cast<float16, float>(alpha).x;
        half beta_fp16;
        beta_fp16.x = dragon_cast<float16, float>(beta).x;
        CUBLAS_CHECK(cublasHgemm(cublas_handle(), 
                                 cuTransA, CUBLAS_OP_N,
                                 m, 1, k, 
                                 &alpha_fp16, 
                                 reinterpret_cast<const half*>(A), LDA, 
                                 reinterpret_cast<const half*>(x), k, 
                                 &beta_fp16,
                                 reinterpret_cast<half*>(y), LDC));
    } else {
            LOG(FATAL) << "unsupported math type";
    }
}

}    // namespace math

}    // namespace dragon

#endif // WITH_CUDA