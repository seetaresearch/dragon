#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/types.h"

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
_AxpbyInt(const int N, const T alpha, const T beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = alpha * x[i] + beta * y[i];
  }
}

template <typename T>
__global__ void
_AxpbyFloat(const int N, const T alpha, const T beta, const T* x, T* y) {
  const auto madd = math::FMAFunctor<T>();
  const auto mul = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = madd(alpha, x[i], mul(beta, y[i]));
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
DEFINE_COPY_FUNC(bfloat16);
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
DEFINE_COPY_FUNC(bfloat16);
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
DEFINE_COPY_MATRIX_FUNC(bfloat16);
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
    } else if (x != y) {                                                       \
      CUDA_CHECK(cudaMemcpyAsync(                                              \
          y, x, sizeof(T) * N, cudaMemcpyDeviceToDevice, ctx->cuda_stream())); \
    }                                                                          \
  }

DEFINE_SCALE_FUNC(uint8_t);
DEFINE_SCALE_FUNC(int8_t);
DEFINE_SCALE_FUNC(int);
DEFINE_SCALE_FUNC(int64_t);
#undef DEFINE_SCALE_FUNC

#define DEFINE_SCALE_FUNC(T, DataType, ExecType)                               \
  template <>                                                                  \
  DRAGON_API void Scale<T, CUDAContext>(                                       \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) {    \
    if (x != y) {                                                              \
      CUDA_CHECK(cudaMemcpyAsync(                                              \
          y, x, sizeof(T) * N, cudaMemcpyDeviceToDevice, ctx->cuda_stream())); \
    }                                                                          \
    if (alpha != 1.f) {                                                        \
      const auto& handle = ctx->cublas_handle();                               \
      CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));    \
      CUBLAS_CHECK(cublasScalEx(                                               \
          handle, N, &alpha, CUDA_R_32F, y, DataType, 1, ExecType));           \
    }                                                                          \
  }

DEFINE_SCALE_FUNC(float16, CUDA_R_16F, CUDA_R_32F);
DEFINE_SCALE_FUNC(bfloat16, CUDA_R_16BF, CUDA_R_32F);
DEFINE_SCALE_FUNC(float, CUDA_R_32F, CUDA_R_32F);
DEFINE_SCALE_FUNC(double, CUDA_R_64F, CUDA_R_64F);
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

#define DEFINE_AXPY_FUNC(T, DataType, ExecType)                             \
  template <>                                                               \
  DRAGON_API void Axpy<T, CUDAContext>(                                     \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    const auto& handle = ctx->cublas_handle();                              \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));   \
    CUBLAS_CHECK(cublasAxpyEx(                                              \
        handle,                                                             \
        N,                                                                  \
        &alpha,                                                             \
        CUDA_R_32F,                                                         \
        x,                                                                  \
        DataType,                                                           \
        1,                                                                  \
        y,                                                                  \
        DataType,                                                           \
        1,                                                                  \
        ExecType));                                                         \
  }

DEFINE_AXPY_FUNC(float16, CUDA_R_16F, CUDA_R_32F);
DEFINE_AXPY_FUNC(bfloat16, CUDA_R_16BF, CUDA_R_32F);
DEFINE_AXPY_FUNC(float, CUDA_R_32F, CUDA_R_32F);
DEFINE_AXPY_FUNC(double, CUDA_R_64F, CUDA_R_64F);
#undef DEFINE_AXPY_FUNC

#define DEFINE_AXPBY_FUNC(T, Kernel)                                           \
  template <>                                                                  \
  DRAGON_API void Axpby<T, CUDAContext>(                                       \
      const int N,                                                             \
      const float alpha,                                                       \
      const T* x,                                                              \
      const float beta,                                                        \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    if ((N & 1) == 0 && math::Traits<T>::HasPack2()) {                         \
      _##Kernel<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N >> 1,                                                              \
          convert::To<math::Traits<T>::scalar2_type>(alpha),                   \
          convert::To<math::Traits<T>::scalar2_type>(beta),                    \
          reinterpret_cast<const math::Traits<T>::scalar2_type*>(x),           \
          reinterpret_cast<math::Traits<T>::scalar2_type*>(y));                \
    } else {                                                                   \
      _##Kernel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
          N,                                                                   \
          convert::To<math::Traits<T>::scalar_type>(alpha),                    \
          convert::To<math::Traits<T>::scalar_type>(beta),                     \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(x),            \
          reinterpret_cast<math::Traits<T>::scalar_type*>(y));                 \
    }                                                                          \
  }

DEFINE_AXPBY_FUNC(int8_t, AxpbyInt);
DEFINE_AXPBY_FUNC(uint8_t, AxpbyInt);
DEFINE_AXPBY_FUNC(int, AxpbyInt);
DEFINE_AXPBY_FUNC(int64_t, AxpbyInt);
DEFINE_AXPBY_FUNC(float16, AxpbyFloat);
DEFINE_AXPBY_FUNC(bfloat16, AxpbyFloat);
DEFINE_AXPBY_FUNC(float, AxpbyFloat);
DEFINE_AXPBY_FUNC(double, AxpbyFloat);
#undef DEFINE_AXPBY_FUNC

#define DEFINE_DOT_FUNC(T, DataType, ExecType)                                 \
  template <>                                                                  \
  DRAGON_API void Dot<T, CUDAContext>(                                         \
      const int N, const T* a, const T* b, T* y, CUDAContext* ctx) {           \
    const auto& handle = ctx->cublas_handle();                                 \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));    \
    CUBLAS_CHECK(cublasDotEx(                                                  \
        handle, N, a, DataType, 1, b, DataType, 1, y, DataType, ExecType));    \
  }                                                                            \
  template <>                                                                  \
  DRAGON_API T Dot<T, CUDAContext>(                                            \
      const int N, const T* a, const T* b, CUDAContext* ctx) {                 \
    T ret;                                                                     \
    const auto& handle = ctx->cublas_handle();                                 \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));      \
    CUBLAS_CHECK(cublasDotEx(                                                  \
        handle, N, a, DataType, 1, b, DataType, 1, &ret, DataType, ExecType)); \
    return ret;                                                                \
  }

DEFINE_DOT_FUNC(float16, CUDA_R_16F, CUDA_R_32F);
DEFINE_DOT_FUNC(bfloat16, CUDA_R_16BF, CUDA_R_32F);
DEFINE_DOT_FUNC(float, CUDA_R_32F, CUDA_R_32F);
DEFINE_DOT_FUNC(double, CUDA_R_64F, CUDA_R_64F);
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

#define DEFINE_GEMV_FUNC(T, DataType, ExecType)                               \
  template <>                                                                 \
  DRAGON_API void Gemv<T, CUDAContext>(                                       \
      const CBLAS_TRANSPOSE TransA,                                           \
      const int M,                                                            \
      const int N,                                                            \
      const float alpha,                                                      \
      const T* A,                                                             \
      const T* x,                                                             \
      const float beta,                                                       \
      T* y,                                                                   \
      CUDAContext* ctx) {                                                     \
    const auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N; \
    const int m = cuTransA == CUBLAS_OP_N ? N : M;                            \
    const int k = cuTransA == CUBLAS_OP_N ? M : N;                            \
    const int lda = cuTransA == CUBLAS_OP_N ? m : k, ldc = m;                 \
    const auto alpha64f = double(alpha), beta64f = double(beta);              \
    const auto& handle = ctx->cublas_handle();                                \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));     \
    CUBLAS_CHECK(cublasGemmEx(                                                \
        handle,                                                               \
        cuTransA,                                                             \
        CUBLAS_OP_N,                                                          \
        m,                                                                    \
        1,                                                                    \
        k,                                                                    \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&alpha64f : &alpha,          \
        A,                                                                    \
        DataType,                                                             \
        lda,                                                                  \
        x,                                                                    \
        DataType,                                                             \
        k,                                                                    \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&beta64f : &beta,            \
        y,                                                                    \
        DataType,                                                             \
        ldc,                                                                  \
        ExecType,                                                             \
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));                                      \
  }

DEFINE_GEMV_FUNC(float16, CUDA_R_16F, CUBLAS_COMPUTE_32F);
DEFINE_GEMV_FUNC(bfloat16, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
DEFINE_GEMV_FUNC(float, CUDA_R_32F, CUBLAS_COMPUTE_32F);
DEFINE_GEMV_FUNC(double, CUDA_R_64F, CUBLAS_COMPUTE_64F);
#undef DEFINE_GEMV_FUNC

#define DEFINE_GEMM_FUNC(T, DataType, ExecType)                               \
  template <>                                                                 \
  DRAGON_API void Gemm<T, CUDAContext>(                                       \
      const CBLAS_TRANSPOSE TransA,                                           \
      const CBLAS_TRANSPOSE TransB,                                           \
      const int M,                                                            \
      const int N,                                                            \
      const int K,                                                            \
      const float alpha,                                                      \
      const T* A,                                                             \
      const T* B,                                                             \
      const float beta,                                                       \
      T* C,                                                                   \
      CUDAContext* ctx) {                                                     \
    const auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; \
    const auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; \
    const int lda = TransA == CblasNoTrans ? K : M;                           \
    const int ldb = TransB == CblasNoTrans ? N : K;                           \
    const auto alpha64f = double(alpha), beta64f = double(beta);              \
    const auto& handle = ctx->cublas_handle();                                \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));     \
    CUBLAS_CHECK(cublasGemmEx(                                                \
        handle,                                                               \
        cuTransB,                                                             \
        cuTransA,                                                             \
        N,                                                                    \
        M,                                                                    \
        K,                                                                    \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&alpha64f : &alpha,          \
        B,                                                                    \
        DataType,                                                             \
        ldb,                                                                  \
        A,                                                                    \
        DataType,                                                             \
        lda,                                                                  \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&beta64f : &beta,            \
        C,                                                                    \
        DataType,                                                             \
        N,                                                                    \
        ExecType,                                                             \
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));                                      \
  }

DEFINE_GEMM_FUNC(float16, CUDA_R_16F, CUBLAS_COMPUTE_32F);
DEFINE_GEMM_FUNC(bfloat16, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
DEFINE_GEMM_FUNC(float, CUDA_R_32F, CUBLAS_COMPUTE_32F);
DEFINE_GEMM_FUNC(double, CUDA_R_64F, CUBLAS_COMPUTE_64F);
#undef DEFINE_GEMM_FUNC

#define DEFINE_BATCHED_GEMM_FUNC(T, DataType, ExecType)                       \
  template <>                                                                 \
  DRAGON_API void GemmBatched<T, CUDAContext>(                                \
      const CBLAS_TRANSPOSE TransA,                                           \
      const CBLAS_TRANSPOSE TransB,                                           \
      const int batch_size,                                                   \
      const int M,                                                            \
      const int N,                                                            \
      const int K,                                                            \
      const float alpha,                                                      \
      const T** A,                                                            \
      const T** B,                                                            \
      const float beta,                                                       \
      T** C,                                                                  \
      CUDAContext* ctx) {                                                     \
    const auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; \
    const auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; \
    const int lda = TransA == CblasNoTrans ? K : M;                           \
    const int ldb = TransB == CblasNoTrans ? N : K;                           \
    const int ldc = N;                                                        \
    const auto alpha64f = double(alpha), beta64f = double(beta);              \
    thrust::device_vector<const void*> A_arr(A, A + batch_size);              \
    thrust::device_vector<const void*> B_arr(B, B + batch_size);              \
    thrust::device_vector<void*> C_arr(C, C + batch_size);                    \
    const auto& handle = ctx->cublas_handle();                                \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));     \
    CUBLAS_CHECK(cublasGemmBatchedEx(                                         \
        handle,                                                               \
        cuTransB,                                                             \
        cuTransA,                                                             \
        N,                                                                    \
        M,                                                                    \
        K,                                                                    \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&alpha64f : &alpha,          \
        B_arr.data().get(),                                                   \
        DataType,                                                             \
        ldb,                                                                  \
        A_arr.data().get(),                                                   \
        DataType,                                                             \
        lda,                                                                  \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&beta64f : &beta,            \
        C_arr.data().get(),                                                   \
        DataType,                                                             \
        ldc,                                                                  \
        batch_size,                                                           \
        ExecType,                                                             \
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));                                      \
  }

DEFINE_BATCHED_GEMM_FUNC(float16, CUDA_R_16F, CUBLAS_COMPUTE_32F);
DEFINE_BATCHED_GEMM_FUNC(bfloat16, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
DEFINE_BATCHED_GEMM_FUNC(float, CUDA_R_32F, CUBLAS_COMPUTE_32F);
DEFINE_BATCHED_GEMM_FUNC(double, CUDA_R_64F, CUBLAS_COMPUTE_64F);
#undef DEFINE_BATCHED_GEMM_FUNC

#define DEFINE_STRIDED_BATCHED_GEMM_FUNC(T, DataType, ExecType)               \
  template <>                                                                 \
  DRAGON_API void GemmStridedBatched<T, CUDAContext>(                         \
      const CBLAS_TRANSPOSE TransA,                                           \
      const CBLAS_TRANSPOSE TransB,                                           \
      const int batch_size,                                                   \
      const int M,                                                            \
      const int N,                                                            \
      const int K,                                                            \
      const int A_stride,                                                     \
      const int B_stride,                                                     \
      const int C_stride,                                                     \
      const float alpha,                                                      \
      const T* A,                                                             \
      const T* B,                                                             \
      const float beta,                                                       \
      T* C,                                                                   \
      CUDAContext* ctx) {                                                     \
    const auto cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; \
    const auto cuTransB = TransB == CblasNoTrans ? CUBLAS_OP_N : CUBLAS_OP_T; \
    const int lda = TransA == CblasNoTrans ? K : M;                           \
    const int ldb = TransB == CblasNoTrans ? N : K;                           \
    const int ldc = N;                                                        \
    const auto alpha64f = double(alpha), beta64f = double(beta);              \
    const auto& handle = ctx->cublas_handle();                                \
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));     \
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(                                  \
        handle,                                                               \
        cuTransB,                                                             \
        cuTransA,                                                             \
        N,                                                                    \
        M,                                                                    \
        K,                                                                    \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&alpha64f : &alpha,          \
        B,                                                                    \
        DataType,                                                             \
        ldb,                                                                  \
        B_stride,                                                             \
        A,                                                                    \
        DataType,                                                             \
        lda,                                                                  \
        A_stride,                                                             \
        ExecType == CUBLAS_COMPUTE_64F ? (float*)&beta64f : &beta,            \
        C,                                                                    \
        DataType,                                                             \
        ldc,                                                                  \
        C_stride,                                                             \
        batch_size,                                                           \
        ExecType,                                                             \
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));                                      \
  }

DEFINE_STRIDED_BATCHED_GEMM_FUNC(float16, CUDA_R_16F, CUBLAS_COMPUTE_32F);
DEFINE_STRIDED_BATCHED_GEMM_FUNC(bfloat16, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
DEFINE_STRIDED_BATCHED_GEMM_FUNC(float, CUDA_R_32F, CUBLAS_COMPUTE_32F);
DEFINE_STRIDED_BATCHED_GEMM_FUNC(double, CUDA_R_64F, CUBLAS_COMPUTE_64F);
#undef DEFINE_STRIDED_BATCHED_GEMM_FUNC

} // namespace math

} // namespace dragon
