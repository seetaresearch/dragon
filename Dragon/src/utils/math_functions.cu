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
    CUDA_POST_KERNEL_CHECK;
}

template <> void Set<int, CUDAContext>(const int n,
                                       const int alpha,
                                       int* x) {
    if (alpha == 0) {
        CUDA_CHECK(cudaMemset(x, 0, sizeof(int) * n));
        return;
    }
    _Set<int> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, alpha, x);
    CUDA_POST_KERNEL_CHECK;
}

template <> void RandomUniform<uint32_t, CUDAContext>(const int n,
                                                      const float low,
                                                      const float high,
                                                      uint32_t* x) {
    //  note that we ignore the low / high
    //  curand could only generates in the range of [0, uint32]
    CURAND_CHECK(curandGenerate(curand_generator(), x, n));
}

template <> void RandomNormal<float, CUDAContext>(const int n,
                                                  const float mu,
                                                  const float sigma,
                                                  float* x) {
    CURAND_CHECK(curandGenerateNormal(curand_generator(), x, n, mu, sigma));
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
    CUDA_POST_KERNEL_CHECK;
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
    CUDA_POST_KERNEL_CHECK;
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
    CUDA_POST_KERNEL_CHECK;
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
    CUDA_POST_KERNEL_CHECK;
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
    CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void _Exp(const int n, const T* a, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = std::exp(a[idx]);
    }
}

template <> void Exp<float, CUDAContext>(int n, const float* x, float *y) {
    _Exp<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, x, y);
    CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void _Log(const int n, const T* a, T* y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = std::log(a[idx]);
    }
}

template <> void Log<float, CUDAContext>(int n, const float* x, float *y) {
    _Log<float> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, x, y);
    CUDA_POST_KERNEL_CHECK;
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

/******************** Level-2 ********************/

template <> void Scal<float, CUDAContext>(const int n, const float alpha, float* y) {
    CUBLAS_CHECK(cublasSscal_v2(cublas_handle(), 
                                n, 
                                &alpha, 
                                y, 1));
}

template <> void Scale<float, CUDAContext>(const int n,
                                           const float alpha,
                                           const float* x,
                                           float* y) {
    CUBLAS_CHECK(cublasScopy_v2(cublas_handle(), n, x, 1, y, 1));
    CUBLAS_CHECK(cublasSscal_v2(cublas_handle(), n, &alpha, y, 1));
}

template <> float StridedDot<float, CUDAContext>(const int n,
                                                 const float* a,
                                                 const int incx,
                                                 const float* b,
                                                 const int incy) {
    float result;
    CUBLAS_CHECK(cublasSdot_v2(cublas_handle(), n,
                                          a, incx,
                                          b, incy,
                                        &result));
    return result;
}

template <> float Dot<float, CUDAContext>(int n, const float* a, const float* b) {
    return StridedDot<float, CUDAContext>(n, a, 1, b, 1);
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
    CUBLAS_CHECK(cublasSaxpy_v2(cublas_handle(), n,
                                      &alpha, x, 1,
                                            y, 1));
}

template <> void Axpby<float, CUDAContext>(const int n,
                                           float alpha,
                                           const float* x,
                                           float beta,
                                           float* y) {
    Scal<float, CUDAContext>(n, beta, y);
    Axpy<float, CUDAContext>(n, alpha, x, y);
}

template <> void RandomUniform<float, CUDAContext>(const int n,
                                                   const float low,
                                                   const float high,
                                                   float* x) {
    CURAND_CHECK(curandGenerateUniform(curand_generator(), x, n));
    float range = high - low;
    if (range != float(1)) Scal<float, CUDAContext>(n, range, x);
    if (low != float(0)) AddScalar<float, CUDAContext>(n, low, x);
}

/******************** Level-3 ********************/

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
    const float _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasSgemm_v2(cublas_handle(),
                             cuTransB, cuTransA,
                                        N, M, K,
                                       &_alpha_,
                                         B, ldb,
                                         A, lda,
                                        &_beta_,
                                         C, N));
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
    const float _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasSgemv_v2(cublas_handle(),
                                       cuTransA,
                                           N, M,
                                       &_alpha_,
                                           A, N,
                                           x, 1,
                                        &_beta_,
                                         y, 1));
}

}    // namespace math

}    // namespace dragon

#endif // WITH_CUDA