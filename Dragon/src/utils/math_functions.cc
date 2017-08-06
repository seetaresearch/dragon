#include <random>

#include "core/context.h"
#include "utils/math_functions.h"

#ifdef WITH_SSE
#include "utils/sse_alternative.h"
#endif

namespace dragon {

namespace math {

/******************** Level-0 ********************/

template <> void Set<float, CPUContext>(const int n, 
                                        const float alpha, 
                                        float* x) {
    if (alpha == 0) {
        memset(x, 0, sizeof(float) * n);
        return;
    }
#ifdef WITH_SSE
    sse::Set<float>(n, alpha, x);
#else   // naive implement
    for (int i = 0; i < n; ++i) x[i] = alpha;
#endif 
}

template <> void Set<int, CPUContext>(const int n, 
                                      const int alpha, 
                                      int* x) {
    if (alpha == 0) {
        memset(x, 0, sizeof(int) * n);
        return;
    }
#ifdef WITH_SSE
    sse::Set<int>(n, alpha, x);
#else  // naive implement
    for (int i = 0; i < n; ++i) x[i] = alpha;
#endif 
}

template <> void Set<float16, CPUContext>(const int n, 
                                          const float16 alpha, 
                                          float16* x) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void RandomUniform<float, CPUContext>(const int n, 
                                                  const float low, 
                                                  const float high, 
                                                  float* x) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (int i = 0; i < n; ++i) {
        x[i] = distribution(*rand_generator());
    }
}

template <> void RandomUniform<float16, CPUContext>(const int n, 
                                                    const float low, 
                                                    const float high, 
                                                    float16* x) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void RandomUniform<uint32_t, CPUContext>(const int n, 
                                                     const float low, 
                                                     const float high, 
                                                     uint32_t* x) {
    std::uniform_int_distribution<uint32_t> distribution(low, high);
    for (int i = 0; i < n; ++i) {
        x[i] = distribution(*rand_generator());
    }
}

template <> void RandomNormal<float, CPUContext>(const int n, 
                                                 const float mu, 
                                                 const float sigma, 
                                                 float* x) {
    std::normal_distribution<float> distribution(mu, sigma);
    for (int i = 0; i < n; ++i) {
        x[i] = distribution(*rand_generator());
    }    
}

template <> void RandomNormal<float16, CPUContext>(const int n, 
                                                   const float mu, 
                                                   const float sigma, 
                                                   float16* x) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void RandomTruncatedNormal<float, CPUContext>(const int n, 
                                                          const float mu, 
                                                          const float sigma,
                                                          const float low,
                                                          const float high,
                                                          float* x) {
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

template <> void RandomTruncatedNormal<float16, CPUContext>(const int n, 
                                                            const float mu,
                                                            const float sigma,
                                                            const float low,
                                                            const float high,
                                                            float16* x) {
    NOT_IMPLEMENTED;
}

template <> void RandomBernoulli<float, CPUContext>(const int n, 
                                                    const float p,
                                                    uint32_t* x) {
    std::bernoulli_distribution distribution(p);
    for (int i = 0; i < n; ++i) {
        x[i] = distribution(*rand_generator());
    }    
}

/******************** Level-1 ********************/

template <> void Add<float, CPUContext>(const int n, 
                                        const float* a, 
                                        const float* b,
                                        float* y) {
#ifdef WITH_SSE
    sse::Add<float>(n, a, b, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
#endif
}

template <> void Sub<float, CPUContext>(const int n, 
                                        const float* a, 
                                        const float* b,
                                        float* y) {
#ifdef WITH_SSE
    sse::Sub<float>(n, a, b, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] = a[i] - b[i];
#endif
}

template <> void Mul<float, CPUContext>(const int n, 
                                        const float* a, 
                                        const float* b,
                                        float* y) {
#ifdef WITH_SSE
    sse::Mul<float>(n, a, b, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] = a[i] * b[i];
#endif
}

template <> void Mul<float16, CPUContext>(const int n, 
                                          const float16* a, 
                                          const float16* b,
                                          float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Div<float, CPUContext>(const int n, 
                                        const float* a, 
                                        const float* b,
                                        float* y) {
#ifdef WITH_SSE
    sse::Div<float>(n, a, b, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] = a[i] / b[i];
#endif    
}

template <> void Div<float16, CPUContext>(const int n, 
                                        const float16* a, 
                                        const float16* b,
                                        float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Clip<float, CPUContext>(const int n, 
                                         const float low, 
                                         const float high,
                                         float* x) {
    for (int i = 0; i < n; ++i) {
        x[i] = std::max(low, std::min(x[i], high));
    }
}

template <> void Exp<float, CPUContext>(int n, 
                                        const float* x, 
                                        float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = std::exp(x[i]);
    }
}

template <> void Log<float, CPUContext>(int n,
                                        const float* x, 
                                        float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = std::log(x[i]);
    }
}

template <> void Square<float, CPUContext>(int n,
                                           const float* x,
                                           float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * x[i];
    }
}

template <> void Square<float16, CPUContext>(int n,
                                             const float16* x,
                                             float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Sqrt<float, CPUContext>(int n,
                                         const float* x,
                                         float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = std::sqrt(x[i]);
    }
}

template <> void Sqrt<float16, CPUContext>(int n,
                                           const float16* x,
                                           float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Pow<float, CPUContext>(int n, 
                                        const float alpha, 
                                        const float* x,
                                        float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = std::pow(x[i], alpha);
    }    
}

template <> void Pow<float16, CPUContext>(int n, 
                                          const float alpha, 
                                          const float16* x,
                                          float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Inv<float, CPUContext>(const int n,
                                        const float numerator,
                                        const float* x, 
                                        float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 1.0 / y[i];
    }
}

template <> void Inv<float16, CPUContext>(const int n,
                                          const float numerator,
                                          const float16* x,
                                          float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

/******************** Level-2 ********************/

template <> void Scal<float, CPUContext>(const int n, 
                                         const float alpha, 
                                         float* y) {
#ifdef WITH_BLAS
    cblas_sscal(n, alpha, y, 1);
#elif  WITH_SSE
    sse::Scal<float>(n, alpha, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] = y[i] * alpha;
#endif
}

template <> void Scal<float16, CPUContext>(const int n, 
                                           const float alpha, 
                                           float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Scale<float16, CPUContext>(const int n, 
                                            const float alpha, 
                                            const float16* x,
                                            float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Scale<float, CPUContext>(const int n, 
                                          const float alpha, 
                                          const float* x,
                                          float* y) {
#ifdef WITH_BLAS
    cblas_scopy(n, x, 1, y, 1);
    cblas_sscal(n, alpha, y, 1);
#elif  WITH_SSE
    sse::Scale<float>(n, alpha, x, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] = x[i] * alpha;
#endif
}

template <> float StridedDot<float, CPUContext>(const int n, 
                                                const float* a, 
                                                const int incx,
                                                const float* b,
                                                const int incy) {
#ifdef WITH_BLAS
    return cblas_sdot(n, a, incx, b, incy);
#else  // naive implement
    float ret = 0.f;
    for (int i = 0; i < n; ++i) ret += a[i] * b[i];
    return ret;
#endif
}

template <> float Dot<float, CPUContext>(int n, 
                                         const float* a, 
                                         const float* b) {
#ifdef WITH_BLAS
    return StridedDot<float, CPUContext>(n, a, 1, b, 1);
#elif  WITH_SSE
    return sse::Dot<float>(n, a, b);
#else  // naive implement
    float ret = 0.f;
    for (int i = 0; i < n; ++i) ret += a[i] * b[i];
    return ret;
#endif
}

template <> float Dot<float16, CPUContext>(int n, 
                                           const float16* a, 
                                           const float16* b) {
    LOG(FATAL) << "unsupport float16 with CPU";
    return 0;
}

template <> float ASum<float, CPUContext>(const int n, const float* x) {
#ifdef WITH_BLAS
    return cblas_sasum(n, x, 1);
#elif  WITH_SSE
    return sse::ASum<float>(n, x);
#else   // naive implement
    float ret = 0.f;
    for (int i = 0; i < n; ++i) ret += x[i];
    return ret;
#endif
}

template <> void AddScalar<float, CPUContext>(const int n,
                                              const float alpha, 
                                              float* y) {
#ifdef  WITH_SSE
    sse::AddScalar<float>(n, alpha, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] += alpha;
#endif
}

template <> void AddScalar<float16, CPUContext>(const int n, 
                                                const float alpha, 
                                                float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void MulScalar<float, CPUContext>(const int n, 
                                              const float alpha,
                                              float* y) {
#ifdef  WITH_SSE
    sse::MulScalar<float>(n, alpha, y);
#else    // naive implement
    for (int i = 0; i < n; ++i) y[i] *= alpha;
#endif
}

template <> void Axpy<float, CPUContext>(const int n, 
                                         float alpha,
                                         const float* x,
                                         float* y) {
#ifdef WITH_BLAS
    cblas_saxpy(n, alpha, x, 1, y, 1);
#elif  WITH_SSE
    sse::Axpy<float>(n, alpha, x, y);
#else  // naive implement
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
#endif
}

template <> void Axpy<float16, CPUContext>(const int n, 
                                           float alpha,
                                           const float16* x,
                                           float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

template <> void Axpby<float, CPUContext>(const int n, 
                                          float alpha, 
                                          const float* x,
                                          float beta,
                                          float *y) {
#ifdef WITH_BLAS
    cblas_sscal(n, beta, y, 1);
    cblas_saxpy(n, alpha, x, 1, y, 1);
#elif  WITH_SSE
    sse::Axpby<float>(n, alpha, x, beta, y);
#else   // naive implement
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + beta* y[i];
#endif
}

template <> void Axpby<float16, CPUContext>(const int n, 
                                            float alpha, 
                                            const float16* x,
                                            float beta,
                                            float16* y) {
    LOG(FATAL) << "unsupport float16 with CPU";
}

/******************** Level-3 ********************/

template <> void Gemm<float, CPUContext>(const CBLAS_TRANSPOSE transA, 
                                         const CBLAS_TRANSPOSE transB,
                                         const int M,
                                         const int N,
                                         const int K,
                                         const float alpha,
                                         const float* A,
                                         const float* B,
                                         const float beta,
                                         float* C,
                                         TensorProto_DataType math_type) {
#ifdef WITH_BLAS
    int lda = (transA == CblasNoTrans) ? K : M;
    int ldb = (transB == CblasNoTrans) ? N : K;
    cblas_sgemm(CblasRowMajor, 
                transA, transB, 
                M, N, K, 
                alpha, 
                A, lda, 
                B, ldb, 
                beta, 
                C, N);
#else    // WITH_BLAS
    LOG(FATAL) << "GEMM with CPU requires BLAS library";
#endif
}

template <> void Gemm<float16, CPUContext>(const CBLAS_TRANSPOSE transA, 
                                           const CBLAS_TRANSPOSE transB,
                                           const int M,
                                           const int N,
                                           const int K,
                                           const float alpha,
                                           const float16* A,
                                           const float16* B,
                                           const float beta,
                                           float16* C,
                                           TensorProto_DataType math_type) {
    LOG(FATAL) << "GEMM with CPU unsupport float16";
}

template <> void Gemv<float, CPUContext>(const CBLAS_TRANSPOSE transA, 
                                         const int M, 
                                         const int N,
                                         const float alpha,
                                         const float* A,
                                         const float* x,
                                         const float beta,
                                         float* y,
                                         TensorProto_DataType math_type) {
#ifdef WITH_BLAS
    int lda = (transA == CblasNoTrans) ? N : M;
    cblas_sgemv(CblasRowMajor, 
                transA, 
                M, N, 
                alpha, 
                A, N, 
                x, 1, 
                beta, 
                y, 1);
#else    // WITH_BLAS
    LOG(FATAL) << "GEMV with CPU requires BLAS library";
#endif
}

template <> void Gemv<float16, CPUContext>(const CBLAS_TRANSPOSE transA, 
                                           const int M, 
                                           const int N,
                                           const float alpha,
                                           const float16* A,
                                           const float16* x,
                                           const float beta,
                                           float16* y,
                                           TensorProto_DataType math_type) {
    LOG(FATAL) << "GEMV with CPU unsupport float16";
}
 
}    // namespace math

}    // namespace dragon