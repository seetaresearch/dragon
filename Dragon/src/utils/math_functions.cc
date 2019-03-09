#include "core/context.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/eigen_utils.h"
#include "utils/math_functions.h"

namespace dragon {

namespace math {

/*!
 * ----------------------------------------------
 *
 *
 *            Simple Unary Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_SIMPLE_UNARY_FUNC(name, T, expr) \
    template <> void name<T, CPUContext>( \
        const int           n, \
        const T*            x, \
        T*                  y, \
        CPUContext*         ctx) { \
        EigenVectorArrayMap<T>(y, n) = \
            ConstEigenVectorArrayMap<T>(x, n).expr(); \
    }

DEFINE_SIMPLE_UNARY_FUNC(Exp, float, exp);
DEFINE_SIMPLE_UNARY_FUNC(Exp, double, exp);
DEFINE_SIMPLE_UNARY_FUNC(Log, float, log);
DEFINE_SIMPLE_UNARY_FUNC(Log, double, log);
DEFINE_SIMPLE_UNARY_FUNC(Inv, float, inverse);
DEFINE_SIMPLE_UNARY_FUNC(Inv, double, inverse);
DEFINE_SIMPLE_UNARY_FUNC(Sqrt, float, sqrt);
DEFINE_SIMPLE_UNARY_FUNC(Sqrt, double, sqrt);
DEFINE_SIMPLE_UNARY_FUNC(RSqrt, float, rsqrt);
DEFINE_SIMPLE_UNARY_FUNC(RSqrt, double, rsqrt);
DEFINE_SIMPLE_UNARY_FUNC(Square, int8_t, square);
DEFINE_SIMPLE_UNARY_FUNC(Square, uint8_t, square);
DEFINE_SIMPLE_UNARY_FUNC(Square, int, square);
DEFINE_SIMPLE_UNARY_FUNC(Square, int64_t, square);
DEFINE_SIMPLE_UNARY_FUNC(Square, float, square);
DEFINE_SIMPLE_UNARY_FUNC(Square, double, square);
#undef DEFINE_SIMPLE_UNARY_FUNC

/*!
 * ----------------------------------------------
 *
 *
 *             Scale Unary Functions
 *
 *
 * ----------------------------------------------
 */

/*!                y = a                 */

#define DEFINE_SET_FUNC(T) \
    template <> void Set<T, CPUContext>( \
        const int               n, \
        const T                 alpha, \
        T*                      y, \
        CPUContext*             ctx) { \
        if (n == 0) return; \
        if (alpha == (T)0) { \
            memset(y, 0, sizeof(T) * n); \
        } else { \
            EigenVectorMap<T>(y, n).setConstant(alpha); \
        } \
    }

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

/*!                y = x^e                */

#define DEFINE_POWX_FUNC(T) \
    template <> void Pow<T, CPUContext>( \
        const int               n, \
        const float             exp, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        EigenVectorArrayMap<T>(y, n) = \
            ConstEigenVectorArrayMap<T>(x, n).pow((T)exp); \
    }

DEFINE_POWX_FUNC(float);
DEFINE_POWX_FUNC(double);
#undef DEFINE_POWX_FUNC

/*!        y = ax    ||    x = ax        */

#define DEFINE_SCALE_FUNC(T) \
    template <> void Scale<T, CPUContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        EigenVectorArrayMap<T>(y, n) = \
            ConstEigenVectorArrayMap<T>(x, n) * (T)alpha; \
    }

DEFINE_SCALE_FUNC(int8_t);
DEFINE_SCALE_FUNC(uint8_t);
DEFINE_SCALE_FUNC(int);
DEFINE_SCALE_FUNC(int64_t);
DEFINE_SCALE_FUNC(float);
DEFINE_SCALE_FUNC(double);
#undef DEFINE_SCALE_FUNC

/*!                y += ax                */

#define DEFINE_AXPY_FUNC(T) \
    template <> void Axpy<T, CPUContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        EigenVectorArrayMap<T>(y, n) += \
            ConstEigenVectorArrayMap<T>(x, n) * (T)alpha; \
    }

DEFINE_AXPY_FUNC(int8_t);
DEFINE_AXPY_FUNC(uint8_t);
DEFINE_AXPY_FUNC(int);
DEFINE_AXPY_FUNC(int64_t);
DEFINE_AXPY_FUNC(float);
DEFINE_AXPY_FUNC(double);
#undef DEFINE_AXPY_FUNC

/*!                 y = ax + by               */

#define DEFINE_AXPBY_FUNC(T) \
    template <> void Axpby<T, CPUContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        const float             beta, \
        T*                      y, \
        CPUContext*            ctx) { \
        Scale(n, beta, y, y, ctx); \
        Axpy(n, alpha, x, y, ctx); \
    }

DEFINE_AXPBY_FUNC(int8_t);
DEFINE_AXPBY_FUNC(uint8_t);
DEFINE_AXPBY_FUNC(int);
DEFINE_AXPBY_FUNC(int64_t);
DEFINE_AXPBY_FUNC(float16);
DEFINE_AXPBY_FUNC(float);
DEFINE_AXPBY_FUNC(double);
#undef DEFINE_AXPBY_FUNC

/*!                 y += a                */

#define DEFINE_ADD_SCALAR_FUNC(T) \
    template <> void AddScalar<T, CPUContext>( \
        const int               n, \
        const float             alpha, \
        T*                      y, \
        CPUContext*             ctx) { \
        T _alpha_ = (T)alpha; \
        if (_alpha_ == T(0)) return; \
        EigenVectorArrayMap<T>(y, n) = \
            ConstEigenVectorArrayMap<T>(y, n) + _alpha_; \
    }

DEFINE_ADD_SCALAR_FUNC(int8_t);
DEFINE_ADD_SCALAR_FUNC(uint8_t);
DEFINE_ADD_SCALAR_FUNC(int);
DEFINE_ADD_SCALAR_FUNC(int64_t);
DEFINE_ADD_SCALAR_FUNC(float);
DEFINE_ADD_SCALAR_FUNC(double);

/*!
 * ----------------------------------------------
 *
 *
 *             Extended Unary Functions
 *
 *
 * ----------------------------------------------
 */

/*!           y = 1 / sqrt(x + eps)          */

#define DEFINE_INVSTD_FUNC(T) \
    template <> void InvStd<T, CPUContext>( \
        const int               n, \
        const float             eps, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        EigenVectorArrayMap<T>(y, n) = \
            (ConstEigenVectorArrayMap<T>(x, n) + (T)eps).rsqrt(); \
    }

DEFINE_INVSTD_FUNC(float);
DEFINE_INVSTD_FUNC(double);
#undef DEFINE_INVSTD_FUNC

/*!                y = sum(x)               */

#define DEFINE_SUM_FUNC(T) \
    template <> void Sum<T, CPUContext>( \
        const int               n, \
        const float             scale, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        T val = ConstEigenVectorArrayMap<T>(x, n).sum(); \
        *y = val * scale; \
    } \
    template <> T Sum<T, CPUContext>( \
        const int               n, \
        const float             scale, \
        const T*                x, \
        CPUContext*             ctx) { \
        T val = ConstEigenVectorArrayMap<T>(x, n).sum(); \
        return val * scale; \
    }

DEFINE_SUM_FUNC(int8_t);
DEFINE_SUM_FUNC(uint8_t);
DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

/*!                y = sum(abs(x)               */

#define DEFINE_ASUM_FUNC(T) \
    template <> T ASum<T, CPUContext>( \
        const int               n, \
        const T*                x, \
        CPUContext*             ctx) { \
        return ConstEigenVectorArrayMap<T>(x, n).abs().sum(); \
    }

DEFINE_ASUM_FUNC(float);
DEFINE_ASUM_FUNC(double);

/*!
 * ----------------------------------------------
 *
 *
 *            Simply Binary Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_SIMPLE_BINARY_FUNC(name, T, expr) \
    template <> \
    void name<T, CPUContext>( \
        const int           n, \
        const T*            a, \
        const T*            b, \
        T*                  y, \
        CPUContext*         ctx) { \
        EigenVectorArrayMap<T>(y, n) = \
            ConstEigenVectorArrayMap<T>(a, n) expr \
                ConstEigenVectorArrayMap<T>(b, n); \
    }

DEFINE_SIMPLE_BINARY_FUNC(Add, int8_t, +);
DEFINE_SIMPLE_BINARY_FUNC(Add, uint8_t, +);
DEFINE_SIMPLE_BINARY_FUNC(Add, int, +);
DEFINE_SIMPLE_BINARY_FUNC(Add, int64_t, +);
DEFINE_SIMPLE_BINARY_FUNC(Add, float, +);
DEFINE_SIMPLE_BINARY_FUNC(Add, double, +);
DEFINE_SIMPLE_BINARY_FUNC(Sub, int8_t, -);
DEFINE_SIMPLE_BINARY_FUNC(Sub, uint8_t, -);
DEFINE_SIMPLE_BINARY_FUNC(Sub, int, -);
DEFINE_SIMPLE_BINARY_FUNC(Sub, int64_t, -);
DEFINE_SIMPLE_BINARY_FUNC(Sub, float, -);
DEFINE_SIMPLE_BINARY_FUNC(Sub, double, -);
DEFINE_SIMPLE_BINARY_FUNC(Mul, int8_t, *);
DEFINE_SIMPLE_BINARY_FUNC(Mul, uint8_t, *);
DEFINE_SIMPLE_BINARY_FUNC(Mul, int, *);
DEFINE_SIMPLE_BINARY_FUNC(Mul, int64_t, *);
DEFINE_SIMPLE_BINARY_FUNC(Mul, float, *);
DEFINE_SIMPLE_BINARY_FUNC(Mul, double, *);
DEFINE_SIMPLE_BINARY_FUNC(Div, int8_t, /);
DEFINE_SIMPLE_BINARY_FUNC(Div, uint8_t, /);
DEFINE_SIMPLE_BINARY_FUNC(Div, int, /);
DEFINE_SIMPLE_BINARY_FUNC(Div, int64_t, /);
DEFINE_SIMPLE_BINARY_FUNC(Div, float, /);
DEFINE_SIMPLE_BINARY_FUNC(Div, double, / );
#undef DEFINE_SIMPLE_BINARY_FUNC

#define DEFINE_DOT_FUNC(T) \
    template <> void Dot<T, CPUContext>( \
        int                     n, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx) { \
            *y = ConstEigenVectorMap<T>(a, n).dot( \
                ConstEigenVectorMap<T>(b, n)); \
    }

DEFINE_DOT_FUNC(float);
DEFINE_DOT_FUNC(double);
#undef DEFINE_DOT_FUNC

/*!
 * ----------------------------------------------
 *
 *
 *          Broadcast Binary Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_BROADCAST_BINARY_FUNCTOR(name) \
    template <typename T, bool BroadcastA> \
    void _RowBroadcast##name( \
        const int               rows, \
        const int               cols, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx); \
    template <typename T, bool BroadcastA> \
    void _ColBroadcast##name( \
        const int               rows, \
        const int               cols, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx); \

#define DEFINE_BROADCAST_BINARY_FUNC(name, T, expr) \
    /*!        y = a + b    ||    a += b        */  \
    template<> void _RowBroadcast##name<T, false>(  \
        const int               rows, \
        const int               cols, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        if (a == y) { \
            EigenArrayMap<T>(y, cols, rows).colwise() expr## = \
                ConstEigenVectorArrayMap<T>(b, cols); \
        } else { \
            EigenArrayMap<T>(y, cols, rows) = \
                ConstEigenArrayMap<T>(a, cols, rows).colwise() expr \
                    ConstEigenVectorArrayMap<T>(b, cols); \
        } \
    } \
    /*!        y = a + b    ||    a += b        */  \
    template<> void _ColBroadcast##name<T, false>(  \
        const int               rows, \
        const int               cols, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        if (a == y) { \
            EigenArrayMap<T>(y, cols, rows).rowwise() expr## = \
                ConstEigenVectorArrayMap<T>(b, rows).transpose(); \
        } else { \
            EigenArrayMap<T>(y, cols, rows) = \
                ConstEigenArrayMap<T>(a, cols, rows).rowwise() expr \
                    ConstEigenVectorArrayMap<T>(b, rows).transpose(); \
        } \
    } \
    /*!        y = a + b    ||    b += a        */  \
    template<> void _RowBroadcast##name<T, true>(  \
        const int               rows, \
        const int               cols, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        if (b == y) { \
            EigenArrayMap<T>(y, cols, rows).colwise() expr## = \
                ConstEigenVectorArrayMap<T>(a, cols); \
        } else { \
            EigenArrayMap<T>(y, cols, rows) = \
                ConstEigenArrayMap<T>(b, cols, rows).colwise() expr \
                    ConstEigenVectorArrayMap<T>(a, cols); \
        } \
    } \
    /*!        y = a + b    ||    b += a        */  \
    template<> void _ColBroadcast##name<T, true>(  \
        const int               rows, \
        const int               cols, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        if (b == y) { \
            EigenArrayMap<T>(y, cols, rows).rowwise() expr## = \
                ConstEigenVectorArrayMap<T>(a, rows).transpose(); \
        } else { \
            EigenArrayMap<T>(y, cols, rows) = \
                ConstEigenArrayMap<T>(b, cols, rows).rowwise() expr \
                    ConstEigenVectorArrayMap<T>(a, rows).transpose(); \
        } \
    } \
    template <> void Broadcast##name<T, CPUContext>( \
        const int               rows, \
        const int               cols, \
        const int               type, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        if (type == 0) { \
            /*! Row - BroadcastB */ \
            _RowBroadcast##name<T, false>(rows, cols, a, b, y, ctx); \
        } else if (type == 1) { \
            /*! Col - BroadcastB */ \
            _ColBroadcast##name<T, false>(rows, cols, a, b, y, ctx); \
        } else if (type == 2) { \
            /*! Row - BroadcastA */ \
            _RowBroadcast##name<T, true> (rows, cols, a, b, y, ctx); \
        } else if (type == 3) { \
            /*! Col - BroadcastA */ \
            _ColBroadcast##name<T, true>(rows, cols, a, b, y, ctx); \
        } else { \
            LOG(FATAL) << "Unknown broadcast type: " << type; \
        } \
    }

DEFINE_BROADCAST_BINARY_FUNCTOR(Add);
DEFINE_BROADCAST_BINARY_FUNCTOR(Sub);
DEFINE_BROADCAST_BINARY_FUNCTOR(Mul);
DEFINE_BROADCAST_BINARY_FUNCTOR(Div);
#undef DEFINE_BROADCAST_BINARY_FUNCTOR

DEFINE_BROADCAST_BINARY_FUNC(Add, int8_t, +);
DEFINE_BROADCAST_BINARY_FUNC(Add, uint8_t, +);
DEFINE_BROADCAST_BINARY_FUNC(Add, int, +);
DEFINE_BROADCAST_BINARY_FUNC(Add, int64_t, +);
DEFINE_BROADCAST_BINARY_FUNC(Add, float, +);
DEFINE_BROADCAST_BINARY_FUNC(Add, double, +);
DEFINE_BROADCAST_BINARY_FUNC(Sub, int8_t, -);
DEFINE_BROADCAST_BINARY_FUNC(Sub, uint8_t, -);
DEFINE_BROADCAST_BINARY_FUNC(Sub, int, -);
DEFINE_BROADCAST_BINARY_FUNC(Sub, int64_t, -);
DEFINE_BROADCAST_BINARY_FUNC(Sub, float, -);
DEFINE_BROADCAST_BINARY_FUNC(Sub, double, -);
DEFINE_BROADCAST_BINARY_FUNC(Mul, int8_t, *);
DEFINE_BROADCAST_BINARY_FUNC(Mul, uint8_t, *);
DEFINE_BROADCAST_BINARY_FUNC(Mul, int, *);
DEFINE_BROADCAST_BINARY_FUNC(Mul, int64_t, *);
DEFINE_BROADCAST_BINARY_FUNC(Mul, float, *);
DEFINE_BROADCAST_BINARY_FUNC(Mul, double, *);
DEFINE_BROADCAST_BINARY_FUNC(Div, int8_t, /);
DEFINE_BROADCAST_BINARY_FUNC(Div, uint8_t, /);
DEFINE_BROADCAST_BINARY_FUNC(Div, int, /);
DEFINE_BROADCAST_BINARY_FUNC(Div, int64_t, /);
DEFINE_BROADCAST_BINARY_FUNC(Div, float, /);
DEFINE_BROADCAST_BINARY_FUNC(Div, double, /);
#undef DEFINE_BROADCAST_BINARY_FUNC

/*!
 * ----------------------------------------------
 *
 *
 *        Linear Algebra Binary Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_GEMM_FUNC(T) \
    template <> void Gemm<T, CPUContext>( \
        const CBLAS_TRANSPOSE   TransA, \
        const CBLAS_TRANSPOSE   TransB, \
        const int               M, \
        const int               N, \
        const int               K, \
        const float             alpha, \
        const T*                A, \
        const T*                B, \
        const float             beta, \
        T*                      C, \
        CPUContext*             ctx, \
        TensorProto_DataType    math_type) { \
        T _alpha_ = alpha, _beta_ = beta; \
        auto C_mat = EigenMatrixMap<T>(C, N, M); \
        if (beta == 0.f) C_mat.setZero(); \
        else C_mat *= _beta_; \
        switch (TransA) { \
            case CblasNoTrans: { \
                switch (TransB) { \
                    case CblasNoTrans: \
                        C_mat.noalias() += _alpha_ * \
                            (ConstEigenMatrixMap<T>(B, N, K) * \
                                ConstEigenMatrixMap<T>(A, K, M)); \
                        return; \
                    case CblasTrans: \
                        C_mat.noalias() += _alpha_ * \
                            (ConstEigenMatrixMap<T>(B, K, N).transpose() * \
                                ConstEigenMatrixMap<T>(A, K, M)); \
                        return; \
                    default: \
                        LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB"; \
                } \
            } \
            case CblasTrans: { \
                switch (TransB) { \
                    case CblasNoTrans: \
                        C_mat.noalias() += _alpha_ * \
                            (ConstEigenMatrixMap<T>(B, N, K) * \
                                ConstEigenMatrixMap<T>(A, M, K).transpose()); \
                        return; \
                    case CblasTrans: \
                        C_mat.noalias() += _alpha_ * \
                            (ConstEigenMatrixMap<T>(B, K, N).transpose() * \
                                ConstEigenMatrixMap<T>(A, M, K).transpose()); \
                        return; \
                    default: \
                        LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB"; \
                } \
            } \
            default: \
                LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransA"; \
        } \
    }

DEFINE_GEMM_FUNC(float);
DEFINE_GEMM_FUNC(double);
#undef DEFINE_GEMM_FUNC

#define DEFINE_GEMV_FUNC(T) \
    template <> void Gemv<T, CPUContext>( \
        const CBLAS_TRANSPOSE   TransA, \
        const int               M, \
        const int               N, \
        const float             alpha, \
        const T*                A, \
        const T*                x, \
        const float             beta, \
        T*                      y, \
        CPUContext*             ctx, \
        TensorProto_DataType    math_type) { \
        T _alpha_ = alpha, _beta_ = beta; \
        EigenVectorMap<T> y_vec(y, TransA == CblasNoTrans ? M : N); \
        if (beta == 0.f) y_vec.setZero(); \
        else y_vec *= _beta_; \
        switch (TransA) { \
            case CblasNoTrans: { \
                y_vec.noalias() += alpha * \
                    (ConstEigenMatrixMap<T>(A, N, M).transpose() * \
                        ConstEigenVectorMap<T>(x, N)); \
                return; \
            } \
            case CblasTrans: { \
                y_vec.noalias() += alpha * \
                    (ConstEigenMatrixMap<T>(A, N, M) * \
                        ConstEigenVectorMap<T>(x, M)); \
                return; \
            } \
            default: \
                LOG(FATAL) << "Gemv float found an unexpected CBLAS_TRANSPOSE input."; \
        } \
    }

DEFINE_GEMV_FUNC(float);
DEFINE_GEMV_FUNC(double);
#undef DEFINE_GEMV_FUNC

/*!
 * ----------------------------------------------
 *
 *
 *               Random Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_RANDOM_UNIFORM_FUNC(T, key) \
    template <> void RandomUniform<T, CPUContext>( \
        const int               n, \
        const float             low, \
        const float             high, \
        T*                      y, \
        CPUContext*             ctx) { \
        std::uniform_##key##_distribution<T> distribution(low, high); \
        auto* rng = ctx->rand_generator(); \
        for (int i = 0; i < n; ++i) y[i] = distribution(*rng); \
    }

#define DEFINE_RANDOM_NORMAL_FUNC(T) \
    template <> void RandomNormal<T, CPUContext>( \
        const int               n, \
        const float             mu, \
        const float             sigma, \
        T*                      y, \
        CPUContext*             ctx) { \
        std::normal_distribution<T> distribution(mu, sigma); \
        auto* rng = ctx->rand_generator(); \
        for (int i = 0; i < n; ++i) y[i] = distribution(*rng); \
    }

#define DEFINE_RANDOM_TRUNCATED_NORMAL_FUNC(T) \
    template <> void RandomTruncatedNormal<T, CPUContext>( \
        const int               n, \
        const float             mu, \
        const float             sigma, \
        const float             low, \
        const float             high, \
        T*                      y, \
        CPUContext*             ctx) { \
        std::normal_distribution<T> distribution(mu, sigma); \
        auto* rng = ctx->rand_generator(); \
        int cur_pos = 0; T gen_value; \
        while (1) { \
            gen_value = distribution(*rng); \
            if (gen_value < low) continue; \
            if (gen_value > high) continue; \
            y[cur_pos++] = gen_value; \
            if (cur_pos >= n) break; \
        } \
    }

#define DEFINE_RANDOM_BERNOULI_FUNC(T) \
    template <> void RandomBernoulli<T, CPUContext>( \
        const int               n, \
        const float             p, \
        T*                      y, \
        CPUContext*             ctx) { \
        std::bernoulli_distribution distribution(p); \
        auto* rng = ctx->rand_generator(); \
        for (int i = 0; i < n; ++i) y[i] = distribution(*rng); \
    }

DEFINE_RANDOM_UNIFORM_FUNC(uint32_t, int);
DEFINE_RANDOM_UNIFORM_FUNC(float, real);
DEFINE_RANDOM_UNIFORM_FUNC(double, real);
#undef DEFINE_RANDOM_UNIFORM_FUNC

DEFINE_RANDOM_NORMAL_FUNC(float);
DEFINE_RANDOM_NORMAL_FUNC(double);
#undef DEFINE_RANDOM_NORMAL_FUNC

DEFINE_RANDOM_TRUNCATED_NORMAL_FUNC(float);
DEFINE_RANDOM_TRUNCATED_NORMAL_FUNC(double);
#undef DEFINE_RANDOM_TRUNCATED_NORMAL_FUNC

DEFINE_RANDOM_BERNOULI_FUNC(uint8_t);
DEFINE_RANDOM_BERNOULI_FUNC(uint32_t);
#undef DEFINE_RANDOM_BERNOULI_FUNC

}  // namespace math

}  // namespace dragon