#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math/blas.h"

namespace dragon {

namespace math {

namespace {

template <typename T>
__global__ void _Scale(const int N, const T alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i] * alpha;
  }
}

template <typename T>
__global__ void _Axpy(const int N, const T alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] += (alpha * x[i]);
  }
}

template <typename T>
__global__ void
_Axpby(const int N, const T alpha, const T* x, const T beta, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = (alpha * x[i] + beta * y[i]);
  }
}

template <>
__global__ void _Axpby<half>(
    const int N,
    const half alpha,
    const half* x,
    const half beta,
    half* y) {
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __hfma(alpha, x[i], __hmul(beta, y[i]));
  }
#else
  const float alpha_val = __half2float(alpha);
  const float beta_val = __half2float(beta);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(
        fmaf(alpha_val, __half2float(x[i]), beta_val * __half2float(y[i])));
  }
#endif
}

template <>
__global__ void _Axpby<half2>(
    const int N,
    const half2 alpha,
    const half2* x,
    const half2 beta,
    half2* y) {
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __hfma2(alpha, x[i], __hmul2(beta, y[i]));
  }
#else
  const float2 alpha_val = __half22float2(alpha);
  const float2 beta_val = __half22float2(beta);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 v1 = __half22float2(x[i]);
    const float2 v2 = __half22float2(y[i]);
    y[i] = __floats2half2_rn(
        fmaf(alpha_val.x, v1.x, beta_val.x * v2.x),
        fmaf(alpha_val.y, v1.y, beta_val.y * v2.y));
  }
#endif
}

template <>
__global__ void _Axpby<float>(
    const int N,
    const float alpha,
    const float* x,
    const float beta,
    float* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = fmaf(alpha, x[i], beta * y[i]);
  }
}

template <>
__global__ void _Axpby<double>(
    const int N,
    const double alpha,
    const double* x,
    const double beta,
    double* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = fma(alpha, x[i], beta * y[i]);
  }
}

} // namespace

#define DEFINE_COPY_FUNC(T)                                                  \
  template <>                                                                \
  DRAGON_API void Copy<T, CUDAContext>(                                      \
      const int N, const T* x, T* y, CUDAContext* ctx) {                     \
    if (N <= 0 || x == y) return;                                            \
    CUDA_CHECK(cudaMemcpyAsync(                                              \
        y, x, N * sizeof(T), cudaMemcpyDeviceToDevice, ctx->cuda_stream())); \
  }

DEFINE_COPY_FUNC(bool);
DEFINE_COPY_FUNC(int8_t);
DEFINE_COPY_FUNC(uint8_t);
DEFINE_COPY_FUNC(int);
DEFINE_COPY_FUNC(int64_t);
DEFINE_COPY_FUNC(float16);
DEFINE_COPY_FUNC(float);
DEFINE_COPY_FUNC(double);
#undef DEFINE_COPY_FUNC

#define DEFINE_COPY_FUNC(T)                   \
  template <>                                 \
  DRAGON_API void Copy<T, CUDAContext>(       \
      const int N,                            \
      const int x_offset,                     \
      const int y_offset,                     \
      const T* x,                             \
      T* y,                                   \
      CUDAContext* ctx) {                     \
    Copy(N, x + x_offset, y + y_offset, ctx); \
  }

DEFINE_COPY_FUNC(bool);
DEFINE_COPY_FUNC(int8_t);
DEFINE_COPY_FUNC(uint8_t);
DEFINE_COPY_FUNC(int);
DEFINE_COPY_FUNC(int64_t);
DEFINE_COPY_FUNC(float16);
DEFINE_COPY_FUNC(float);
DEFINE_COPY_FUNC(double);
#undef DEFINE_COPY_FUNC

#define DEFINE_COPY_MATRIX_FUNC(T)            \
  template <>                                 \
  DRAGON_API void CopyMatrix<T, CUDAContext>( \
      const int M,                            \
      const int N,                            \
      const int ldx,                          \
      const int ldy,                          \
      const int x_offset,                     \
      const int y_offset,                     \
      const T* x,                             \
      T* y,                                   \
      CUDAContext* ctx) {                     \
    if (M <= 0 || N <= 0) return;             \
    CUDA_CHECK(cudaMemcpy2DAsync(             \
        y + y_offset,                         \
        sizeof(T) * ldy,                      \
        x + x_offset,                         \
        sizeof(T) * ldx,                      \
        sizeof(T) * N,                        \
        M,                                    \
        cudaMemcpyDeviceToDevice,             \
        ctx->cuda_stream()));                 \
  }

DEFINE_COPY_MATRIX_FUNC(bool);
DEFINE_COPY_MATRIX_FUNC(int8_t);
DEFINE_COPY_MATRIX_FUNC(uint8_t);
DEFINE_COPY_MATRIX_FUNC(int);
DEFINE_COPY_MATRIX_FUNC(int64_t);
DEFINE_COPY_MATRIX_FUNC(float16);
DEFINE_COPY_MATRIX_FUNC(float);
DEFINE_COPY_MATRIX_FUNC(double);
#undef DEFINE_COPY_MATRIX_FUNC

#define DEFINE_SCALE_FUNC(T)                                                   \
  template <>                                                                  \
  DRAGON_API void Scale<T, CUDAContext>(                                       \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) {    \
    if (alpha != 1.f) {                                                        \
      _Scale<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(         \
          N, static_cast<T>(alpha), x, y);                                     \
      return;                                                                  \
    }                                                                          \
    if (x != y) {                                                              \
      CUDA_CHECK(cudaMemcpyAsync(                                              \
          y, x, sizeof(T) * N, cudaMemcpyDeviceToDevice, ctx->cuda_stream())); \
    }                                                                          \
  }

DEFINE_SCALE_FUNC(uint8_t);
DEFINE_SCALE_FUNC(int8_t);
DEFINE_SCALE_FUNC(int);
DEFINE_SCALE_FUNC(int64_t);
#undef DEFINE_SCALE_FUNC

#define DEFINE_SCALE_FUNC(T, cublasFunc)                                       \
  template <>                                                                  \
  DRAGON_API void Scale<T, CUDAContext>(                                       \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) {    \
    if (x != y) {                                                              \
      CUDA_CHECK(cudaMemcpyAsync(                                              \
          y, x, sizeof(T) * N, cudaMemcpyDeviceToDevice, ctx->cuda_stream())); \
    }                                                                          \
    if (alpha != 1.f) {                                                        \
      T alpha_val = static_cast<T>(alpha);                                     \
      const auto& handle = ctx->cublas_handle();                               \
      CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));    \
      CUBLAS_CHECK(cublasFunc(handle, N, &alpha_val, y, 1));                   \
    }                                                                          \
  }

template <>
DRAGON_API void Scale<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if (x != y) {
    CUDA_CHECK(cudaMemcpyAsync(
        y,
        x,
        sizeof(float16) * N,
        cudaMemcpyDeviceToDevice,
        ctx->cuda_stream()));
  }
  if (alpha != 1.f) {
    const auto& handle = ctx->cublas_handle();
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasScalEx(
        handle, N, &alpha, CUDA_R_32F, y, CUDA_R_16F, 1, CUDA_R_32F));
  }
}

DEFINE_SCALE_FUNC(float, cublasSscal);
DEFINE_SCALE_FUNC(double, cublasDscal);
#undef DEFINE_SCALE_FUNC

#define DEFINE_AXPY_FUNC(T)                                                 \
  template <>                                                               \
  DRAGON_API void Axpy<T, CUDAContext>(                                     \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    _Axpy<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(         \
        N, static_cast<T>(alpha), x, y);                                    \
  }

DEFINE_AXPY_FUNC(int8_t);
DEFINE_AXPY_FUNC(uint8_t);
DEFINE_AXPY_FUNC(int);
DEFINE_AXPY_FUNC(int64_t);
#undef DEFINE_AXPY_FUNC

#define DEFINE_AXPY_FUNC(T, cublasFunc)                                     \
  template <>                                                               \
  DRAGON_API void Axpy<T, CUDAContext>(                                     \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    T alpha_val = static_cast<T>(alpha);                                    \
    const auto& handle = ctx->cublas_handle();                              \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));   \
    CUBLAS_CHECK(cublasFunc(handle, N, &alpha_val, x, 1, y, 1));            \
  }

template <>
DRAGON_API void Axpy<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  const auto& handle = ctx->cublas_handle();
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasAxpyEx(
      handle,
      N,
      &alpha,
      CUDA_R_32F,
      x,
      CUDA_R_16F,
      1,
      y,
      CUDA_R_16F,
      1,
      CUDA_R_32F));
}

DEFINE_AXPY_FUNC(float, cublasSaxpy);
DEFINE_AXPY_FUNC(double, cublasDaxpy);
#undef DEFINE_AXPY_FUNC

#define DEFINE_AXPBY_FUNC(T)                                         \
  template <>                                                        \
  DRAGON_API void Axpby<T, CUDAContext>(                             \
      const int N,                                                   \
      const float alpha,                                             \
      const T* x,                                                    \
      const float beta,                                              \
      T* y,                                                          \
      CUDAContext* ctx) {                                            \
    _Axpby<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, static_cast<T>(alpha), x, static_cast<T>(beta), y);       \
  }

template <>
DRAGON_API void Axpby<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* x,
    const float beta,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _Axpby<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        convert::To<half2>(alpha),
        reinterpret_cast<const half2*>(x),
        convert::To<half2>(beta),
        reinterpret_cast<half2*>(y));
  } else {
    _Axpby<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        convert::To<half>(alpha),
        reinterpret_cast<const half*>(x),
        convert::To<half>(beta),
        reinterpret_cast<half*>(y));
  }
}

DEFINE_AXPBY_FUNC(int8_t);
DEFINE_AXPBY_FUNC(uint8_t);
DEFINE_AXPBY_FUNC(int);
DEFINE_AXPBY_FUNC(int64_t);
DEFINE_AXPBY_FUNC(float);
DEFINE_AXPBY_FUNC(double);
#undef DEFINE_AXPBY_FUNC

#define DEFINE_DOT_FUNC(T, cublasFunc)                                      \
  template <>                                                               \
  DRAGON_API void Dot<T, CUDAContext>(                                      \
      const int N, const T* a, const T* b, T* y, CUDAContext* ctx) {        \
    const auto& handle = ctx->cublas_handle();                              \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE)); \
    CUBLAS_CHECK(cublasFunc(handle, N, a, 1, b, 1, y));                     \
  }                                                                         \
  template <>                                                               \
  DRAGON_API T Dot<T, CUDAContext>(                                         \
      const int N, const T* a, const T* b, CUDAContext* ctx) {              \
    T ret;                                                                  \
    const auto& handle = ctx->cublas_handle();                              \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));   \
    CUBLAS_CHECK(cublasFunc(handle, N, a, 1, b, 1, &ret));                  \
    return ret;                                                             \
  }

template <>
DRAGON_API void Dot<float16, CUDAContext>(
    const int N,
    const float16* a,
    const float16* b,
    float16* y,
    CUDAContext* ctx) {
  const auto& handle = ctx->cublas_handle();
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasDotEx(
      handle,
      N,
      a,
      CUDA_R_16F,
      1,
      b,
      CUDA_R_16F,
      1,
      y,
      CUDA_R_16F,
      CUDA_R_32F));
}

DEFINE_DOT_FUNC(float, cublasSdot);
DEFINE_DOT_FUNC(double, cublasDdot);
#undef DEFINE_DOT_FUNC

#define DEFINE_ASUM_FUNC(T, cublasFunc)                                     \
  template <>                                                               \
  DRAGON_API void ASum<T, CUDAContext>(                                     \
      const int N, const T* x, T* y, CUDAContext* ctx) {                    \
    const auto& handle = ctx->cublas_handle();                              \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE)); \
    CUBLAS_CHECK(cublasFunc(handle, N, x, 1, y));                           \
  }                                                                         \
  template <>                                                               \
  DRAGON_API T ASum<T, CUDAContext>(                                        \
      const int N, const T* x, CUDAContext* ctx) {                          \
    T ret;                                                                  \
    const auto& handle = ctx->cublas_handle();                              \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));   \
    CUBLAS_CHECK(cublasFunc(handle, N, x, 1, &ret));                        \
    return ret;                                                             \
  }

DEFINE_ASUM_FUNC(float, cublasSasum);
DEFINE_ASUM_FUNC(double, cublasDasum);
#undef DEFINE_ASUM_FUNC

template <>
DRAGON_API void Gemv<float16, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float16* A,
    const float16* x,
    const float beta,
    float16* y,
    CUDAContext* ctx) {
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int m = cuTransA == CUBLAS_OP_N ? N : M;
  const int k = cuTransA == CUBLAS_OP_N ? M : N;
  const int LDA = cuTransA == CUBLAS_OP_N ? m : k;
  const int LDC = m;
  const auto& handle = ctx->cublas_handle();
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#endif
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  if (TENSOR_CORE_AVAILABLE()) {
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        cuTransA,
        CUBLAS_OP_N,
        m,
        1,
        k,
        &alpha,
        A,
        CUDA_R_16F,
        LDA,
        x,
        CUDA_R_16F,
        k,
        &beta,
        y,
        CUDA_R_16F,
        LDC,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    CUBLAS_CHECK(cublasSgemmEx(
        handle,
        cuTransA,
        CUBLAS_OP_N,
        m,
        1,
        k,
        &alpha,
        A,
        CUDA_R_16F,
        LDA,
        x,
        CUDA_R_16F,
        k,
        &beta,
        y,
        CUDA_R_16F,
        LDC));
  }
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
}

#define DEFINE_GEMV_FUNC(T, cublasFunc)                                    \
  template <>                                                              \
  DRAGON_API void Gemv<T, CUDAContext>(                                    \
      const CBLAS_TRANSPOSE TransA,                                        \
      const int M,                                                         \
      const int N,                                                         \
      const float alpha,                                                   \
      const T* A,                                                          \
      const T* x,                                                          \
      const float beta,                                                    \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N;    \
    const auto alpha_val = static_cast<T>(alpha);                          \
    const auto beta_val = static_cast<T>(beta);                            \
    const auto& handle = ctx->cublas_handle();                             \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));  \
    CUBLAS_CHECK(cublasFunc(                                               \
        handle, cuTransA, N, M, &alpha_val, A, N, x, 1, &beta_val, y, 1)); \
  }

DEFINE_GEMV_FUNC(float, cublasSgemv);
DEFINE_GEMV_FUNC(double, cublasDgemv);
#undef DEFINE_GEMV_FUNC

template <>
DRAGON_API void Gemm<float16, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float16* A,
    const float16* B,
    const float beta,
    float16* C,
    CUDAContext* ctx) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  const auto& handle = ctx->cublas_handle();
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#endif
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  if (TENSOR_CORE_AVAILABLE()) {
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_16F,
        ldb,
        A,
        CUDA_R_16F,
        lda,
        &beta,
        C,
        CUDA_R_16F,
        N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    CUBLAS_CHECK(cublasSgemmEx(
        handle,
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_16F,
        ldb,
        A,
        CUDA_R_16F,
        lda,
        &beta,
        C,
        CUDA_R_16F,
        N));
  }
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
}

#define DEFINE_GEMM_FUNC(T, cublasFunc)                                   \
  template <>                                                             \
  DRAGON_API void Gemm<T, CUDAContext>(                                   \
      const CBLAS_TRANSPOSE TransA,                                       \
      const CBLAS_TRANSPOSE TransB,                                       \
      const int M,                                                        \
      const int N,                                                        \
      const int K,                                                        \
      const float alpha,                                                  \
      const T* A,                                                         \
      const T* B,                                                         \
      const float beta,                                                   \
      T* C,                                                               \
      CUDAContext* ctx) {                                                 \
    int lda = TransA == CblasNoTrans ? K : M;                             \
    int ldb = TransB == CblasNoTrans ? N : K;                             \
    auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;   \
    auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;   \
    const auto alpha_val = static_cast<T>(alpha);                         \
    const auto beta_val = static_cast<T>(beta);                           \
    const auto& handle = ctx->cublas_handle();                            \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)); \
    CUBLAS_CHECK(cublasFunc(                                              \
        handle,                                                           \
        cuTransB,                                                         \
        cuTransA,                                                         \
        N,                                                                \
        M,                                                                \
        K,                                                                \
        &alpha_val,                                                       \
        B,                                                                \
        ldb,                                                              \
        A,                                                                \
        lda,                                                              \
        &beta_val,                                                        \
        C,                                                                \
        N));                                                              \
  }

DEFINE_GEMM_FUNC(float, cublasSgemm);
DEFINE_GEMM_FUNC(double, cublasDgemm);
#undef DEFINE_GEMM_FUNC

template <>
DRAGON_API void GemmBatched<float16, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float16** A,
    const float16** B,
    const float beta,
    float16** C,
    CUDAContext* ctx) {
  int lda = TransA == CblasNoTrans ? K : M;
  int ldb = TransB == CblasNoTrans ? N : K;
  int ldc = N;
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  thrust::device_vector<const void*> A_arr(A, A + batch_size);
  thrust::device_vector<const void*> B_arr(B, B + batch_size);
  thrust::device_vector<void*> C_arr(C, C + batch_size);
  const auto& handle = ctx->cublas_handle();
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#endif
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasGemmBatchedEx(
      handle,
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha,
      B_arr.data().get(),
      CUDA_R_16F,
      ldb,
      A_arr.data().get(),
      CUDA_R_16F,
      lda,
      &beta,
      C_arr.data().get(),
      CUDA_R_16F,
      ldc,
      batch_size,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
}

#define DEFINE_BATCHED_GEMM_FUNC(T, cublasFunc)                           \
  template <>                                                             \
  DRAGON_API void GemmBatched<T, CUDAContext>(                            \
      const CBLAS_TRANSPOSE TransA,                                       \
      const CBLAS_TRANSPOSE TransB,                                       \
      const int batch_size,                                               \
      const int M,                                                        \
      const int N,                                                        \
      const int K,                                                        \
      const float alpha,                                                  \
      const T** A,                                                        \
      const T** B,                                                        \
      const float beta,                                                   \
      T** C,                                                              \
      CUDAContext* ctx) {                                                 \
    int lda = TransA == CblasNoTrans ? K : M;                             \
    int ldb = TransB == CblasNoTrans ? N : K;                             \
    int ldc = N;                                                          \
    auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;   \
    auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;   \
    const auto alpha_val = static_cast<T>(alpha);                         \
    const auto beta_val = static_cast<T>(beta);                           \
    thrust::device_vector<const T*> A_arr(A, A + batch_size);             \
    thrust::device_vector<const T*> B_arr(B, B + batch_size);             \
    thrust::device_vector<T*> C_arr(C, C + batch_size);                   \
    const auto& handle = ctx->cublas_handle();                            \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)); \
    CUBLAS_CHECK(cublasFunc(                                              \
        handle,                                                           \
        cuTransB,                                                         \
        cuTransA,                                                         \
        N,                                                                \
        M,                                                                \
        K,                                                                \
        &alpha_val,                                                       \
        B_arr.data().get(),                                               \
        ldb,                                                              \
        A_arr.data().get(),                                               \
        lda,                                                              \
        &beta_val,                                                        \
        C_arr.data().get(),                                               \
        ldc,                                                              \
        batch_size));                                                     \
  }

DEFINE_BATCHED_GEMM_FUNC(float, cublasSgemmBatched);
DEFINE_BATCHED_GEMM_FUNC(double, cublasDgemmBatched);
#undef DEFINE_BATCHED_GEMM_FUNC

template <>
DRAGON_API void GemmStridedBatched<float16, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const int A_stride,
    const int B_stride,
    const int C_stride,
    const float alpha,
    const float16* A,
    const float16* B,
    const float beta,
    float16* C,
    CUDAContext* ctx) {
  int lda = TransA == CblasNoTrans ? K : M;
  int ldb = TransB == CblasNoTrans ? N : K;
  int ldc = N;
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  const auto& handle = ctx->cublas_handle();
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#endif
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      handle,
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha,
      B,
      CUDA_R_16F,
      ldb,
      B_stride,
      A,
      CUDA_R_16F,
      lda,
      A_stride,
      &beta,
      C,
      CUDA_R_16F,
      ldc,
      C_stride,
      batch_size,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
}

#define DEFINE_STRIDED_BATCHED_GEMM_FUNC(T, cublasFunc)                   \
  template <>                                                             \
  DRAGON_API void GemmStridedBatched<T, CUDAContext>(                     \
      const CBLAS_TRANSPOSE TransA,                                       \
      const CBLAS_TRANSPOSE TransB,                                       \
      const int batch_size,                                               \
      const int M,                                                        \
      const int N,                                                        \
      const int K,                                                        \
      const int A_stride,                                                 \
      const int B_stride,                                                 \
      const int C_stride,                                                 \
      const float alpha,                                                  \
      const T* A,                                                         \
      const T* B,                                                         \
      const float beta,                                                   \
      T* C,                                                               \
      CUDAContext* ctx) {                                                 \
    int lda = TransA == CblasNoTrans ? K : M;                             \
    int ldb = TransB == CblasNoTrans ? N : K;                             \
    int ldc = N;                                                          \
    auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;   \
    auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;   \
    const auto alpha_val = static_cast<T>(alpha);                         \
    const auto beta_val = static_cast<T>(beta);                           \
    const auto& handle = ctx->cublas_handle();                            \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)); \
    CUBLAS_CHECK(cublasFunc(                                              \
        handle,                                                           \
        cuTransB,                                                         \
        cuTransA,                                                         \
        N,                                                                \
        M,                                                                \
        K,                                                                \
        &alpha_val,                                                       \
        B,                                                                \
        ldb,                                                              \
        B_stride,                                                         \
        A,                                                                \
        lda,                                                              \
        A_stride,                                                         \
        &beta_val,                                                        \
        C,                                                                \
        ldc,                                                              \
        C_stride,                                                         \
        batch_size));                                                     \
  }

DEFINE_STRIDED_BATCHED_GEMM_FUNC(float, cublasSgemmStridedBatched);
DEFINE_STRIDED_BATCHED_GEMM_FUNC(double, cublasDgemmStridedBatched);
#undef DEFINE_STRIDED_BATCHED_GEMM_FUNC

} // namespace math

} // namespace dragon
