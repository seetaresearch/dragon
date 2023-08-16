#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/types.h"

namespace dragon {

namespace math {

#define DEFINE_COPY_FUNC(T)                             \
  template <>                                           \
  DRAGON_API void Copy<T, CPUContext>(                  \
      const int N, const T* x, T* y, CPUContext* ctx) { \
    if (N <= 0 || x == y) return;                       \
    memcpy(y, x, sizeof(T) * N);                        \
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
  DRAGON_API void Copy<T, CPUContext>(        \
      const int N,                            \
      const int x_offset,                     \
      const int y_offset,                     \
      const T* x,                             \
      T* y,                                   \
      CPUContext* ctx) {                      \
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

#define DEFINE_COPY_MATRIX_FUNC(T)                                   \
  template <>                                                        \
  DRAGON_API void CopyMatrix<T, CPUContext>(                         \
      const int M,                                                   \
      const int N,                                                   \
      const int ldx,                                                 \
      const int ldy,                                                 \
      const int x_offset,                                            \
      const int y_offset,                                            \
      const T* x,                                                    \
      T* y,                                                          \
      CPUContext* ctx) {                                             \
    auto* offset_x = x + x_offset;                                   \
    auto* offset_y = y + y_offset;                                   \
    if (M <= 0 || N <= 0) return;                                    \
    if (ldx == N && ldy == N) {                                      \
      if (offset_x != offset_y) {                                    \
        memcpy(offset_y, offset_x, sizeof(T) * M * N);               \
      }                                                              \
      return;                                                        \
    }                                                                \
    for (int i = 0; i < M; ++i) {                                    \
      memcpy(offset_y + ldy * i, offset_x + ldx * i, sizeof(T) * N); \
    }                                                                \
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
  DRAGON_API void Scale<T, CPUContext>(                                        \
      const int N, const float alpha, const T* x, T* y, CPUContext* ctx) {     \
    using EigenT = math::Traits<T>::eigen_type;                                \
    EigenVectorArrayMap<EigenT>((EigenT*)y, N) =                               \
        ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N) * EigenT(alpha); \
  }

DEFINE_SCALE_FUNC(int8_t);
DEFINE_SCALE_FUNC(uint8_t);
DEFINE_SCALE_FUNC(int);
DEFINE_SCALE_FUNC(int64_t);
DEFINE_SCALE_FUNC(float16);
DEFINE_SCALE_FUNC(bfloat16);
DEFINE_SCALE_FUNC(float);
DEFINE_SCALE_FUNC(double);
#undef DEFINE_SCALE_FUNC

#define DEFINE_AXPY_FUNC(T)                                                    \
  template <>                                                                  \
  DRAGON_API void Axpy<T, CPUContext>(                                         \
      const int N, const float alpha, const T* x, T* y, CPUContext* ctx) {     \
    using EigenT = math::Traits<T>::eigen_type;                                \
    EigenVectorArrayMap<EigenT>((EigenT*)y, N) +=                              \
        ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N) * EigenT(alpha); \
  }

DEFINE_AXPY_FUNC(int8_t);
DEFINE_AXPY_FUNC(uint8_t);
DEFINE_AXPY_FUNC(int);
DEFINE_AXPY_FUNC(int64_t);
DEFINE_AXPY_FUNC(float16);
DEFINE_AXPY_FUNC(bfloat16);
DEFINE_AXPY_FUNC(float);
DEFINE_AXPY_FUNC(double);
#undef DEFINE_AXPY_FUNC

#define DEFINE_AXPBY_FUNC(T)            \
  template <>                           \
  DRAGON_API void Axpby<T, CPUContext>( \
      const int N,                      \
      const float alpha,                \
      const T* x,                       \
      const float beta,                 \
      T* y,                             \
      CPUContext* ctx) {                \
    Scale(N, beta, y, y, ctx);          \
    Axpy(N, alpha, x, y, ctx);          \
  }

DEFINE_AXPBY_FUNC(int8_t);
DEFINE_AXPBY_FUNC(uint8_t);
DEFINE_AXPBY_FUNC(int);
DEFINE_AXPBY_FUNC(int64_t);
DEFINE_AXPBY_FUNC(float16);
DEFINE_AXPBY_FUNC(bfloat16);
DEFINE_AXPBY_FUNC(float);
DEFINE_AXPBY_FUNC(double);
#undef DEFINE_AXPBY_FUNC

#define DEFINE_DOT_FUNC(T)                                                 \
  template <>                                                              \
  DRAGON_API void Dot<T, CPUContext>(                                      \
      int N, const T* a, const T* b, T* y, CPUContext* ctx) {              \
    using EigenT = math::Traits<T>::eigen_type;                            \
    auto* y_alias = (EigenT*)y;                                            \
    *y_alias = ConstEigenVectorMap<EigenT>((const EigenT*)a, N)            \
                   .dot(ConstEigenVectorMap<EigenT>((const EigenT*)b, N)); \
  }                                                                        \
  template <>                                                              \
  DRAGON_API T Dot<T, CPUContext>(                                         \
      int N, const T* a, const T* b, CPUContext* ctx) {                    \
    using EigenT = math::Traits<T>::eigen_type;                            \
    auto ret = ConstEigenVectorMap<EigenT>((const EigenT*)a, N)            \
                   .dot(ConstEigenVectorMap<EigenT>((const EigenT*)b, N)); \
    return *((T*)&ret);                                                    \
  }

DEFINE_DOT_FUNC(float16);
DEFINE_DOT_FUNC(bfloat16);
DEFINE_DOT_FUNC(float);
DEFINE_DOT_FUNC(double);
#undef DEFINE_DOT_FUNC

#define DEFINE_ASUM_FUNC(T)                                                    \
  template <>                                                                  \
  DRAGON_API void ASum<T, CPUContext>(                                         \
      const int N, const T* x, T* y, CPUContext* ctx) {                        \
    *y = ConstEigenVectorArrayMap<T>(x, N).abs().sum();                        \
  }                                                                            \
  template <>                                                                  \
  DRAGON_API T ASum<T, CPUContext>(const int N, const T* x, CPUContext* ctx) { \
    return ConstEigenVectorArrayMap<T>(x, N).abs().sum();                      \
  }

DEFINE_ASUM_FUNC(float);
DEFINE_ASUM_FUNC(double);
#undef DEFINE_ASUM_FUNC

#define DEFINE_GEMV_FUNC(T)                                                    \
  template <>                                                                  \
  DRAGON_API void Gemv<T, CPUContext>(                                         \
      const CBLAS_TRANSPOSE TransA,                                            \
      const int M,                                                             \
      const int N,                                                             \
      const float alpha,                                                       \
      const T* A,                                                              \
      const T* x,                                                              \
      const float beta,                                                        \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    using EigenT = math::Traits<T>::eigen_type;                                \
    ConstEigenVectorMap<EigenT> X((const EigenT*)x, N);                        \
    EigenVectorMap<EigenT> Y((EigenT*)y, TransA == CblasNoTrans ? M : N);      \
    ConstEigenMatrixMap<EigenT> A_mat((const EigenT*)A, N, M);                 \
    if (beta == 0.f) {                                                         \
      Y.setZero();                                                             \
    } else {                                                                   \
      Y *= EigenT(beta);                                                       \
    }                                                                          \
    switch (TransA) {                                                          \
      case CblasNoTrans:                                                       \
        Y.noalias() += EigenT(alpha) * (A_mat.transpose() * X);                \
        return;                                                                \
      case CblasTrans:                                                         \
        Y.noalias() += EigenT(alpha) * (A_mat * X);                            \
        return;                                                                \
      default:                                                                 \
        LOG(FATAL) << "Gemv float found an unexpected CBLAS_TRANSPOSE input."; \
    }                                                                          \
  }

DEFINE_GEMV_FUNC(float16);
DEFINE_GEMV_FUNC(bfloat16);
DEFINE_GEMV_FUNC(float);
DEFINE_GEMV_FUNC(double);
#undef DEFINE_GEMV_FUNC

#define DEFINE_GEMM_FUNC(T)                                           \
  template <>                                                         \
  DRAGON_API void Gemm<T, CPUContext>(                                \
      const CBLAS_TRANSPOSE TransA,                                   \
      const CBLAS_TRANSPOSE TransB,                                   \
      const int M,                                                    \
      const int N,                                                    \
      const int K,                                                    \
      const float alpha,                                              \
      const T* A,                                                     \
      const T* B,                                                     \
      const float beta,                                               \
      T* C,                                                           \
      CPUContext* ctx) {                                              \
    using EigenT = math::Traits<T>::eigen_type;                       \
    using ConstMatrixMap = ConstEigenMatrixMap<EigenT>;               \
    EigenMatrixMap<EigenT> C_mat((EigenT*)C, N, M);                   \
    if (beta == 0.f) {                                                \
      C_mat.setZero();                                                \
    } else {                                                          \
      C_mat *= EigenT(beta);                                          \
    }                                                                 \
    switch (TransA) {                                                 \
      case CblasNoTrans: {                                            \
        switch (TransB) {                                             \
          case CblasNoTrans:                                          \
            C_mat.noalias() += EigenT(alpha) *                        \
                (ConstMatrixMap((const EigenT*)B, N, K) *             \
                 ConstMatrixMap((const EigenT*)A, K, M));             \
            return;                                                   \
          case CblasTrans:                                            \
            C_mat.noalias() += EigenT(alpha) *                        \
                (ConstMatrixMap((const EigenT*)B, K, N).transpose() * \
                 ConstMatrixMap((const EigenT*)A, K, M));             \
            return;                                                   \
          default:                                                    \
            LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB.";   \
        }                                                             \
      }                                                               \
      case CblasTrans: {                                              \
        switch (TransB) {                                             \
          case CblasNoTrans:                                          \
            C_mat.noalias() += EigenT(alpha) *                        \
                (ConstMatrixMap((const EigenT*)B, N, K) *             \
                 ConstMatrixMap((const EigenT*)A, M, K).transpose()); \
            return;                                                   \
          case CblasTrans:                                            \
            C_mat.noalias() += EigenT(alpha) *                        \
                (ConstMatrixMap((const EigenT*)B, K, N).transpose() * \
                 ConstMatrixMap((const EigenT*)A, M, K).transpose()); \
            return;                                                   \
          default:                                                    \
            LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB.";   \
        }                                                             \
      }                                                               \
      default:                                                        \
        LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransA.";       \
    }                                                                 \
  }

DEFINE_GEMM_FUNC(float16);
DEFINE_GEMM_FUNC(bfloat16);
DEFINE_GEMM_FUNC(float);
DEFINE_GEMM_FUNC(double);
#undef DEFINE_GEMM_FUNC

#define DEFINE_BATCHED_GEMM_FUNC(T)                                      \
  template <>                                                            \
  DRAGON_API void GemmBatched<T, CPUContext>(                            \
      const CBLAS_TRANSPOSE TransA,                                      \
      const CBLAS_TRANSPOSE TransB,                                      \
      const int batch_size,                                              \
      const int M,                                                       \
      const int N,                                                       \
      const int K,                                                       \
      const float alpha,                                                 \
      const T** A,                                                       \
      const T** B,                                                       \
      const float beta,                                                  \
      T** C,                                                             \
      CPUContext* ctx) {                                                 \
    for (int i = 0; i < batch_size; ++i) {                               \
      Gemm(TransA, TransB, M, N, K, alpha, A[i], B[i], beta, C[i], ctx); \
    }                                                                    \
  }

DEFINE_BATCHED_GEMM_FUNC(float16);
DEFINE_BATCHED_GEMM_FUNC(bfloat16);
DEFINE_BATCHED_GEMM_FUNC(float);
DEFINE_BATCHED_GEMM_FUNC(double);
#undef DEFINE_BATCHED_GEMM_FUNC

#define DEFINE_STRIDED_BATCHED_GEMM_FUNC(T)          \
  template <>                                        \
  DRAGON_API void GemmStridedBatched<T, CPUContext>( \
      const CBLAS_TRANSPOSE TransA,                  \
      const CBLAS_TRANSPOSE TransB,                  \
      const int batch_size,                          \
      const int M,                                   \
      const int N,                                   \
      const int K,                                   \
      const int A_stride,                            \
      const int B_stride,                            \
      const int C_stride,                            \
      const float alpha,                             \
      const T* A,                                    \
      const T* B,                                    \
      const float beta,                              \
      T* C,                                          \
      CPUContext* ctx) {                             \
    for (int i = 0; i < batch_size; ++i) {           \
      Gemm(                                          \
          TransA,                                    \
          TransB,                                    \
          M,                                         \
          N,                                         \
          K,                                         \
          alpha,                                     \
          A + i * A_stride,                          \
          B + i * B_stride,                          \
          beta,                                      \
          C + i * C_stride,                          \
          ctx);                                      \
    }                                                \
  }

DEFINE_STRIDED_BATCHED_GEMM_FUNC(float16);
DEFINE_STRIDED_BATCHED_GEMM_FUNC(bfloat16);
DEFINE_STRIDED_BATCHED_GEMM_FUNC(float);
DEFINE_STRIDED_BATCHED_GEMM_FUNC(double);
#undef DEFINE_STRIDED_BATCHED_GEMM_FUNC

} // namespace math

} // namespace dragon
