#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/conversions.h"
#include "dragon/utils/math/blas.h"

namespace dragon {

namespace math {

namespace {

template <typename T>
__global__ void _Scale(const int n, const T alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = x[i] * alpha;
  }
}

template <typename T>
__global__ void
_Copy(const int n, const int incx, const int incy, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i * incy] = x[i * incx];
  }
}

template <typename T>
__global__ void _Axpy(const int n, const T alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] += (alpha * x[i]);
  }
}

template <typename T>
__global__ void
_Axpby(const int n, const T alpha, const T* x, const T beta, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = (alpha * x[i] + beta * y[i]);
  }
}

template <>
__global__ void _Axpby<half>(
    const int n,
    const half alpha,
    const half* x,
    const half beta,
    half* y) {
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __hfma(alpha, x[i], __hmul(beta, y[i]));
  }
#else
  const float alpha_val = __half2float(alpha);
  const float beta_val = __half2float(beta);
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __float2half(
        fmaf(alpha_val, __half2float(x[i]), beta_val * __half2float(y[i])));
  }
#endif
}

template <>
__global__ void _Axpby<half2>(
    const int n,
    const half2 alpha,
    const half2* x,
    const half2 beta,
    half2* y) {
#if __CUDA_ARCH__ >= 530
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __hfma2(alpha, x[i], __hmul2(beta, y[i]));
  }
#else
  const float2 alpha_val = __half22float2(alpha);
  const float2 beta_val = __half22float2(beta);
  CUDA_1D_KERNEL_LOOP(i, n) {
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
    const int n,
    const float alpha,
    const float* x,
    const float beta,
    float* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = fmaf(alpha, x[i], beta * y[i]);
  }
}

template <>
__global__ void _Axpby<double>(
    const int n,
    const double alpha,
    const double* x,
    const double beta,
    double* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = fma(alpha, x[i], beta * y[i]);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_SCALE_FUNC(T)                                                \
  template <>                                                               \
  DRAGON_API void Scale<T, CUDAContext>(                                    \
      const int n, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    if (alpha != 1.f) {                                                     \
      if (x != y) {                                                         \
        cudaMemcpyAsync(                                                    \
            y,                                                              \
            x,                                                              \
            sizeof(T) * n,                                                  \
            cudaMemcpyDeviceToDevice,                                       \
            ctx->cuda_stream());                                            \
      }                                                                     \
      return;                                                               \
    }                                                                       \
    _Scale<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
        n, static_cast<T>(alpha), x, y);                                    \
  }

DEFINE_SCALE_FUNC(int8_t);
DEFINE_SCALE_FUNC(uint8_t);
DEFINE_SCALE_FUNC(int);
DEFINE_SCALE_FUNC(int64_t);
#undef DEFINE_SCALE_FUNC

#define DEFINE_SCALE_FUNC(T, cublas_func)                                      \
  template <>                                                                  \
  DRAGON_API void Scale<T, CUDAContext>(                                       \
      const int n, const float alpha, const T* x, T* y, CUDAContext* ctx) {    \
    if (x != y) {                                                              \
      CUDA_CHECK(cudaMemcpyAsync(                                              \
          y, x, sizeof(T) * n, cudaMemcpyDeviceToDevice, ctx->cuda_stream())); \
    }                                                                          \
    if (alpha != 1.f) {                                                        \
      T alpha_val = static_cast<T>(alpha);                                     \
      CUBLAS_CHECK(cublasSetPointerMode(                                       \
          ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));                    \
      CUBLAS_CHECK(cublas_func(ctx->cublas_handle(), n, &alpha_val, y, 1));    \
    }                                                                          \
  }

template <>
DRAGON_API void Scale<float16, CUDAContext>(
    const int n,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if (x != y) {
    CUDA_CHECK(cudaMemcpyAsync(
        y,
        x,
        sizeof(float16) * n,
        cudaMemcpyDeviceToDevice,
        ctx->cuda_stream()));
  }
  if (alpha != 1.f) {
    CUBLAS_CHECK(
        cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasScalEx(
        ctx->cublas_handle(),
        n,
        &alpha,
        CUDA_R_32F,
        y,
        CUDA_R_16F,
        1,
        CUDA_R_32F));
  }
}

DEFINE_SCALE_FUNC(float, cublasSscal);
DEFINE_SCALE_FUNC(double, cublasDscal);
#undef DEFINE_SCALE_FUNC

#define DEFINE_COPY_FUNC(T)                                                    \
  template <>                                                                  \
  DRAGON_API void Copy<T, CUDAContext>(                                        \
      const int n, const T* x, T* y, CUDAContext* ctx) {                       \
    if (x != y && n > 0) {                                                     \
      CUDA_CHECK(cudaMemcpyAsync(                                              \
          y, x, n * sizeof(T), cudaMemcpyDeviceToDevice, ctx->cuda_stream())); \
    }                                                                          \
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

#define DEFINE_COPY_FUNC(T)                                           \
  template <>                                                         \
  DRAGON_API void Copy<T, CUDAContext>(                               \
      const int n,                                                    \
      const int incx,                                                 \
      const int incy,                                                 \
      const T* x,                                                     \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    if (x != y && n > 0) {                                            \
      _Copy<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          n, incx, incy, x, y);                                       \
    }                                                                 \
  }

DEFINE_COPY_FUNC(bool);
DEFINE_COPY_FUNC(int8_t);
DEFINE_COPY_FUNC(uint8_t);
DEFINE_COPY_FUNC(int);
DEFINE_COPY_FUNC(int64_t);
DEFINE_COPY_FUNC(float16);
#undef DEFINE_COPY_FUNC

#define DEFINE_COPY_FUNC(T, cublas_func)                                    \
  template <>                                                               \
  DRAGON_API void Copy<T, CUDAContext>(                                     \
      const int n,                                                          \
      const int incx,                                                       \
      const int incy,                                                       \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    if (x != y && n > 0) {                                                  \
      CUBLAS_CHECK(cublas_func(ctx->cublas_handle(), n, x, incx, y, incy)); \
    }                                                                       \
  }

DEFINE_COPY_FUNC(float, cublasScopy);
DEFINE_COPY_FUNC(double, cublasDcopy);
#undef DEFINE_COPY_FUNC

#define DEFINE_COPY_FUNC(T)                   \
  template <>                                 \
  DRAGON_API void CopyMatrix<T, CUDAContext>( \
      const int m,                            \
      const int n,                            \
      const int ldx,                          \
      const int ldy,                          \
      const T* x,                             \
      T* y,                                   \
      CUDAContext* ctx) {                     \
    if (m <= 0 || n <= 0) return;             \
    CUDA_CHECK(cudaMemcpy2DAsync(             \
        y,                                    \
        sizeof(T) * ldy,                      \
        x,                                    \
        sizeof(T) * ldx,                      \
        sizeof(T) * n,                        \
        m,                                    \
        cudaMemcpyDeviceToDevice,             \
        ctx->cuda_stream()));                 \
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

#define DEFINE_AXPY_FUNC(T)                                                 \
  template <>                                                               \
  DRAGON_API void Axpy<T, CUDAContext>(                                     \
      const int n, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    _Axpy<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(         \
        n, static_cast<T>(alpha), x, y);                                    \
  }

DEFINE_AXPY_FUNC(int8_t);
DEFINE_AXPY_FUNC(uint8_t);
DEFINE_AXPY_FUNC(int);
DEFINE_AXPY_FUNC(int64_t);
#undef DEFINE_AXPY_FUNC

#define DEFINE_AXPY_FUNC(T, cublas_func)                                       \
  template <>                                                                  \
  DRAGON_API void Axpy<T, CUDAContext>(                                        \
      const int n, const float alpha, const T* x, T* y, CUDAContext* ctx) {    \
    T alpha_val = static_cast<T>(alpha);                                       \
    CUBLAS_CHECK(                                                              \
        cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST)); \
    CUBLAS_CHECK(                                                              \
        cublas_func(ctx->cublas_handle(), n, &alpha_val, x, 1, y, 1));         \
  }

template <>
DRAGON_API void Axpy<float16, CUDAContext>(
    const int n,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasAxpyEx(
      ctx->cublas_handle(),
      n,
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
      const int n,                                                   \
      const float alpha,                                             \
      const T* x,                                                    \
      const float beta,                                              \
      T* y,                                                          \
      CUDAContext* ctx) {                                            \
    _Axpby<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        n, static_cast<T>(alpha), x, static_cast<T>(beta), y);       \
  }

template <>
DRAGON_API void Axpby<float16, CUDAContext>(
    const int n,
    const float alpha,
    const float16* x,
    const float beta,
    float16* y,
    CUDAContext* ctx) {
  if ((n & 1) == 0) {
    _Axpby<<<CUDA_BLOCKS(n >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n >> 1,
        convert::To<half2>(alpha),
        reinterpret_cast<const half2*>(x),
        convert::To<half2>(beta),
        reinterpret_cast<half2*>(y));
  } else {
    _Axpby<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n,
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

#define DEFINE_DOT_FUNC(T, cublas_func)                                        \
  template <>                                                                  \
  DRAGON_API void Dot<T, CUDAContext>(                                         \
      const int n, const T* a, const T* b, T* y, CUDAContext* ctx) {           \
    CUBLAS_CHECK(cublasSetPointerMode(                                         \
        ctx->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));                    \
    CUBLAS_CHECK(cublas_func(ctx->cublas_handle(), n, a, 1, b, 1, y));         \
  }                                                                            \
  template <>                                                                  \
  DRAGON_API T Dot<T, CUDAContext>(                                            \
      const int n, const T* a, const T* b, CUDAContext* ctx) {                 \
    T ret;                                                                     \
    CUBLAS_CHECK(                                                              \
        cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST)); \
    CUBLAS_CHECK(cublas_func(ctx->cublas_handle(), n, a, 1, b, 1, &ret));      \
    return ret;                                                                \
  }

template <>
DRAGON_API void Dot<float16, CUDAContext>(
    const int n,
    const float16* a,
    const float16* b,
    float16* y,
    CUDAContext* ctx) {
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasDotEx(
      ctx->cublas_handle(),
      n,
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

#define DEFINE_ASUM_FUNC(T, cublas_func)                                       \
  template <>                                                                  \
  DRAGON_API void ASum<T, CUDAContext>(                                        \
      const int n, const T* x, T* y, CUDAContext* ctx) {                       \
    CUBLAS_CHECK(cublasSetPointerMode(                                         \
        ctx->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));                    \
    CUBLAS_CHECK(cublas_func(ctx->cublas_handle(), n, x, 1, y));               \
  }                                                                            \
  template <>                                                                  \
  DRAGON_API T ASum<T, CUDAContext>(                                           \
      const int n, const T* x, CUDAContext* ctx) {                             \
    T ret;                                                                     \
    CUBLAS_CHECK(                                                              \
        cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST)); \
    CUBLAS_CHECK(cublas_func(ctx->cublas_handle(), n, x, 1, &ret));            \
    return ret;                                                                \
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
    CUDAContext* ctx,
    const string math_type) {
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = cuTransA == CUBLAS_OP_N ? N : M;
  int k = cuTransA == CUBLAS_OP_N ? M : N;
  int LDA = cuTransA == CUBLAS_OP_N ? m : k;
  int LDC = m;
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  if (math_type == "float32") {
#if CUDA_VERSION >= 9000
    if (TENSOR_CORE_AVAILABLE()) {
      // GEMV + MATH32 + TENSOR-CORE
      CUBLAS_CHECK(cublasGemmEx(
          ctx->cublas_handle(),
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
      // GEMV + MATH32 + DEFAULT
      CUBLAS_CHECK(cublasSgemmEx(
          ctx->cublas_handle(),
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
#else
    CUBLAS_CHECK(cublasSgemmEx(
        ctx->cublas_handle(),
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
#endif
  } else if (math_type == "float16") {
    const half alpha_val = convert::To<half>(alpha);
    const half beta_val = convert::To<half>(beta);
#if CUDA_VERSION >= 9000
    if (TENSOR_CORE_AVAILABLE()) {
      // GEMV + MATH16 + TENSOR-CORE
      CUBLAS_CHECK(cublasGemmEx(
          ctx->cublas_handle(),
          cuTransA,
          CUBLAS_OP_N,
          m,
          1,
          k,
          &alpha_val,
          A,
          CUDA_R_16F,
          LDA,
          x,
          CUDA_R_16F,
          k,
          &beta_val,
          y,
          CUDA_R_16F,
          LDC,
          CUDA_R_16F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
      // GEMV + MATH16 + DEFAULT
      CUBLAS_CHECK(cublasHgemm(
          ctx->cublas_handle(),
          cuTransA,
          CUBLAS_OP_N,
          m,
          1,
          k,
          &alpha_val,
          reinterpret_cast<const half*>(A),
          LDA,
          reinterpret_cast<const half*>(x),
          k,
          &beta_val,
          reinterpret_cast<half*>(y),
          LDC));
    }
#else
    CUBLAS_CHECK(cublasHgemm(
        ctx->cublas_handle(),
        cuTransA,
        CUBLAS_OP_N,
        m,
        1,
        k,
        &alpha_val,
        reinterpret_cast<const half*>(A),
        LDA,
        reinterpret_cast<const half*>(x),
        k,
        &beta_val,
        reinterpret_cast<half*>(y),
        LDC));
#endif
  } else {
    LOG(FATAL) << "Unknown math type: " << math_type;
  }
}

template <>
DRAGON_API void Gemv<float, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CUDAContext* ctx,
    const string math_type) {
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasSgemv(
      ctx->cublas_handle(), cuTransA, N, M, &alpha, A, N, x, 1, &beta, y, 1));
}

template <>
DRAGON_API void Gemv<double, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const double* A,
    const double* x,
    const float beta,
    double* y,
    CUDAContext* ctx,
    const string math_type) {
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto alpha_val = static_cast<double>(alpha);
  const auto beta_val = static_cast<double>(beta);
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasDgemv(
      ctx->cublas_handle(),
      cuTransA,
      N,
      M,
      &alpha_val,
      A,
      N,
      x,
      1,
      &beta_val,
      y,
      1));
}

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
    CUDAContext* ctx,
    const std::string math_type) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  if (math_type == "float32") {
#if CUDA_VERSION >= 9000
    if (TENSOR_CORE_AVAILABLE()) {
      // GEMM + MATH32 + TENSOR-CORE
      CUBLAS_CHECK(cublasGemmEx(
          ctx->cublas_handle(),
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
      // GEMM + MATH32 + DEFAULT
      CUBLAS_CHECK(cublasSgemmEx(
          ctx->cublas_handle(),
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
#else
    CUBLAS_CHECK(cublasSgemmEx(
        ctx->cublas_handle(),
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
#endif
  } else if (math_type == "float16") {
    const half alpha_val = convert::To<half>(alpha);
    const half beta_val = convert::To<half>(beta);
#if CUDA_VERSION >= 9000
    if (TENSOR_CORE_AVAILABLE()) {
      // GEMM + MATH16 + TENSOR-CORE
      CUBLAS_CHECK(cublasGemmEx(
          ctx->cublas_handle(),
          cuTransB,
          cuTransA,
          N,
          M,
          K,
          &alpha_val,
          B,
          CUDA_R_16F,
          ldb,
          A,
          CUDA_R_16F,
          lda,
          &beta_val,
          C,
          CUDA_R_16F,
          N,
          CUDA_R_16F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
      // GEMM + MATH16 + DEFAULT
      CUBLAS_CHECK(cublasHgemm(
          ctx->cublas_handle(),
          cuTransB,
          cuTransA,
          N,
          M,
          K,
          &alpha_val,
          reinterpret_cast<const half*>(B),
          ldb,
          reinterpret_cast<const half*>(A),
          lda,
          &beta_val,
          reinterpret_cast<half*>(C),
          N));
    }
#else
    CUBLAS_CHECK(cublasHgemm(
        ctx->cublas_handle(),
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        &alpha_val,
        reinterpret_cast<const half*>(B),
        ldb,
        reinterpret_cast<const half*>(A),
        lda,
        &beta_val,
        reinterpret_cast<half*>(C),
        N));
#endif
  } else {
    LOG(FATAL) << "Unknown math type: " << math_type;
  }
}

template <>
DRAGON_API void Gemm<float, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CUDAContext* ctx,
    const string math_type) {
  int lda = TransA == CblasNoTrans ? K : M;
  int ldb = TransB == CblasNoTrans ? N : K;
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasSgemm(
      ctx->cublas_handle(),
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      A,
      lda,
      &beta,
      C,
      N));
}

template <>
DRAGON_API void Gemm<double, CUDAContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const double* A,
    const double* B,
    const float beta,
    double* C,
    CUDAContext* ctx,
    const string math_type) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
  const auto alpha_val = static_cast<double>(alpha);
  const auto beta_val = static_cast<double>(beta);
  CUBLAS_CHECK(
      cublasSetPointerMode(ctx->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasDgemm(
      ctx->cublas_handle(),
      cuTransB,
      cuTransA,
      N,
      M,
      K,
      &alpha_val,
      B,
      ldb,
      A,
      lda,
      &beta_val,
      C,
      N));
}

} // namespace math

} // namespace dragon

#endif // USE_CUDA
