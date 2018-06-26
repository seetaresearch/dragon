#include <random>

#include "core/context.h"
#include "utils/omp_alternative.h"
#include "utils/sse_alternative.h"
#include "utils/math_functions.h"

namespace dragon {

namespace math {

/******************** Level-0 ********************/

template <> void Set<float, CPUContext>(
    const int               n,
    const float             alpha,
    float*                  x) {
    if (alpha == 0) {
        memset(x, 0, sizeof(float) * n);
        return;
    }
#ifdef WITH_SSE
    sse::Set<float>(n, alpha, x);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) x[i] = alpha;
#endif  // WITH_SSE
}

template <> void Set<int, CPUContext>(
    const int               n,
    const int               alpha,
    int*                    x) {
    if (alpha == 0) {
        memset(x, 0, sizeof(int) * n);
        return;
    }
#ifdef WITH_SSE
    sse::Set<int>(n, alpha, x);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) x[i] = alpha;
#endif  // WITH_SSE
}

template <> void Set<float16, CPUContext>(
    const int               n,
    const float16           alpha,
    float16*                x) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void RandomUniform<float, CPUContext>(
    const int               n,
    const float             low,
    const float             high,
    float*                  x) {
    std::uniform_real_distribution<float> distribution(low, high);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) x[i] = distribution(*rand_generator());
}

template <> void RandomUniform<float16, CPUContext>(
    const int               n,
    const float             low,
    const float             high,
    float16*                x) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void RandomUniform<uint32_t, CPUContext>(
    const int               n,
    const float             low,
    const float             high,
    uint32_t*               x) {
    std::uniform_int_distribution<uint32_t> distribution(low, high);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) x[i] = distribution(*rand_generator());
}

template <> void RandomNormal<float, CPUContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    float*                  x) {
    std::normal_distribution<float> distribution(mu, sigma);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) x[i] = distribution(*rand_generator());
}

template <> void RandomNormal<float16, CPUContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    float16*                x) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void RandomTruncatedNormal<float, CPUContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    const float             low,
    const float             high,
    float*                  x) {
    std::normal_distribution<float> distribution(mu, sigma);
    int cur_pos = 0; float gen_value;
    while (1) {
        gen_value = distribution(*rand_generator());
        if (gen_value < low) continue;
        if (gen_value > high) continue;
        x[cur_pos++] = gen_value;
        if (cur_pos >= n) break;
    }
}

template <> void RandomTruncatedNormal<float16, CPUContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    const float             low,
    const float             high,
    float16*                x) {
    NOT_IMPLEMENTED;
}

template <> void RandomBernoulli<float, CPUContext>(
    const int               n,
    const float             p,
    uint32_t*               x) {
    std::bernoulli_distribution distribution(p);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) x[i] = distribution(*rand_generator());
}

/******************** Level-1 ********************/

template <> void Add<float, CPUContext>(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
#ifdef WITH_SSE
    sse::Add<float>(n, a, b, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
#endif  // WITH_SSE
}

template <> void Add<int, CPUContext>(
    const int               n,
    const int*              a,
    const int*              b,
    int*                    y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
}

template <> void Add<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Sub<float, CPUContext>(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
#ifdef WITH_SSE
    sse::Sub<float>(n, a, b, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = a[i] - b[i];
#endif  // WITH_SSE
}

template <> void Sub<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Mul<float, CPUContext>(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
#ifdef WITH_SSE
    sse::Mul<float>(n, a, b, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = a[i] * b[i];
#endif  // WITH_SSE
}

template <> void Mul<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Div<float, CPUContext>(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
#ifdef WITH_SSE
    sse::Div<float>(n, a, b, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = a[i] / b[i];
#endif  // WITH_SSE
}

template <> void Div<float16, CPUContext>(
    const int               n,
    const float16*          a,
    const float16*          b,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Clip<float, CPUContext>(
    const int               n,
    const float             low,
    const float             high,
    float*                  x) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) {
        x[i] = std::max(low, std::min(x[i], high));
    }
}

template <> void Exp<float, CPUContext>(
    int                     n,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = std::exp(x[i]);
}

template <> void Log<float, CPUContext>(
    int                     n,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = std::log(x[i]);
}

template <> void Square<float, CPUContext>(
    int                     n,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = x[i] * x[i];
}

template <> void Square<float16, CPUContext>(
    int                     n,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Sqrt<float, CPUContext>(
    int                     n,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = std::sqrt(x[i]);
}

template <> void Sqrt<float16, CPUContext>(
    int                     n,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Pow<float, CPUContext>(
    int                     n,
    const float             alpha,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = std::pow(x[i], alpha);
}

template <> void Pow<float16, CPUContext>(
    int                     n,
    const float             alpha,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Inv<float, CPUContext>(
    const int               n,
    const float             numerator,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = numerator / y[i];
}

template <> void Inv<float16, CPUContext>(
    const int               n,
    const float             numerator,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** Level-2 ********************/

template <> void Scal<float, CPUContext>(
    const int               n,
    const float             alpha,
    float*                  y) {
#ifdef WITH_BLAS
    cblas_sscal(n, alpha, y, 1);
#elif  WITH_SSE
    sse::Scal<float>(n, alpha, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = y[i] * alpha;
#endif  // WITH_BLAS
}

template <> void Scal<float16, CPUContext>(
    const int               n,
    const float             alpha,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Scale<float16, CPUContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Scale<float, CPUContext>(
    const int               n,
    const float             alpha,
    const float*            x,
    float*                  y) {
#ifdef WITH_BLAS
    cblas_scopy(n, x, 1, y, 1);
    cblas_sscal(n, alpha, y, 1);
#elif  WITH_SSE
    sse::Scale<float>(n, alpha, x, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = x[i] * alpha;
#endif  // WITH_BLAS
}

template <> float StridedDot<float, CPUContext>(
    const int               n,
    const float*            a,
    const int               incx,
    const float*            b,
    const int               incy) {
#ifdef WITH_BLAS
    return cblas_sdot(n, a, incx, b, incy);
#else
    float ret = 0.f;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) ret += a[i] * b[i];
    return ret;
#endif  //  WITH_BLAS
}

template <> float Dot<float, CPUContext>(
    int                     n,
    const float*            a,
    const float*            b) {
#ifdef WITH_BLAS
    return StridedDot<float, CPUContext>(n, a, 1, b, 1);
#elif  WITH_SSE
    return sse::Dot<float>(n, a, b);
#else
    float ret = 0.f;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) ret += a[i] * b[i];
    return ret;
#endif  // WITH_BLAS
}

template <> float Dot<float16, CPUContext>(
    int                     n,
    const float16*          a,
    const float16*          b) {
    CPU_FP16_NOT_SUPPORTED;
    return 0;
}

template <> float ASum<float, CPUContext>(
    const int               n,
    const float*            x) {
#ifdef WITH_BLAS
    return cblas_sasum(n, x, 1);
#elif WITH_SSE
    return sse::ASum<float>(n, x);
#else
    float ret = 0.f;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) ret += x[i];
    return ret;
#endif  // WITH_BLAS
}

template <> void AddScalar<float, CPUContext>(
    const int               n,
    const float             alpha,
    float*                  y) {
#ifdef WITH_SSE
    sse::AddScalar<float>(n, alpha, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] += alpha;
#endif  // WITH_SSE
}

template <> void AddScalar<float16, CPUContext>(
    const int               n,
    const float             alpha,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void MulScalar<float, CPUContext>(
    const int               n,
    const float             alpha,
    float*                  y) {
#ifdef WITH_SSE
    sse::MulScalar<float>(n, alpha, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] *= alpha;
#endif  // WITH_SSE
}

template <> void MulScalar<float16, CPUContext>(
    const int               n,
    const float             alpha,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Axpy<float, CPUContext>(
    const int               n,
    float                   alpha,
    const float*            x,
    float*                  y) {
#ifdef WITH_BLAS
    cblas_saxpy(n, alpha, x, 1, y, 1);
#elif  WITH_SSE
    sse::Axpy<float>(n, alpha, x, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
#endif  // WITH_BLAS
}

template <> void Axpy<float16, CPUContext>(
    const int               n,
    float                   alpha,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Axpby<float, CPUContext>(
    const int               n,
    float                   alpha,
    const float*            x,
    float                   beta,
    float*                  y) {
#ifdef WITH_BLAS
    cblas_sscal(n, beta, y, 1);
    cblas_saxpy(n, alpha, x, 1, y, 1);
#elif  WITH_SSE
    sse::Axpby<float>(n, alpha, x, beta, y);
#else
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(n))
#endif
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + beta* y[i];
#endif  // WITH_BLAS
}

template <> void Axpby<float16, CPUContext>(
    const int               n,
    float                   alpha,
    const float16*          x,
    float                   beta,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** Level-3 ********************/

template <> void Gemm<float, CPUContext>(
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
    TensorProto_DataType    math_type) {
#ifdef WITH_BLAS
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_sgemm(
        CblasRowMajor, TransA, TransB,
            M, N, K, alpha, A, lda, B, ldb, beta, C, N);
#else    // WITH_BLAS
    LOG(FATAL) << "GEMM for CPUContext requires BLAS Library.";
#endif
}

template <> void Gemm<float16, CPUContext>(
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
    TensorProto_DataType    math_type) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Gemv<float, CPUContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const float*            A,
    const float*            x,
    const float             beta,
    float*                  y,
    TensorProto_DataType    math_type) {
#ifdef WITH_BLAS
    int lda = (TransA == CblasNoTrans) ? N : M;
    cblas_sgemv(
        CblasRowMajor, TransA,
            M, N, alpha, A, N, x, 1, beta, y, 1);
#else    // WITH_BLAS
    LOG(FATAL) << "GEMV for CPUContext requires BLAS Library.";
#endif
}

template <> void Gemv<float16, CPUContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const float16*          A,
    const float16*          x,
    const float             beta,
    float16*                y,
    TensorProto_DataType    math_type) {
    CPU_FP16_NOT_SUPPORTED;
}
 
}    // namespace math

}    // namespace dragon