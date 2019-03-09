#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
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

template <typename T>
__global__ void _Exp(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = exp(a[i]);
    }
}

template <typename T>
__global__ void _Log(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = log(a[i]);
    }
}

template <typename T>
__global__ void _Inv(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = (T)1 / x[i];
    }
}

template <typename T>
__global__ void _Sqrt(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = sqrt(x[i]);
    }
}

template <typename T>
__global__ void _RSqrt(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = rsqrt(x[i]);
    }
}

template <typename T>
__global__ void _Square(
    const int               n,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = x[i] * x[i];
    }
}

#define DEFINE_SIMPLE_UNARY_FUNC(name, T) \
    template <> void name<T, CUDAContext>( \
        const int               n, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, x, y); \
    }

DEFINE_SIMPLE_UNARY_FUNC(Exp, float);
DEFINE_SIMPLE_UNARY_FUNC(Exp, double);
DEFINE_SIMPLE_UNARY_FUNC(Log, float);
DEFINE_SIMPLE_UNARY_FUNC(Log, double);
DEFINE_SIMPLE_UNARY_FUNC(Inv, float);
DEFINE_SIMPLE_UNARY_FUNC(Inv, double);
DEFINE_SIMPLE_UNARY_FUNC(Sqrt, float);
DEFINE_SIMPLE_UNARY_FUNC(Sqrt, double);
DEFINE_SIMPLE_UNARY_FUNC(RSqrt, float);
DEFINE_SIMPLE_UNARY_FUNC(RSqrt, double);
DEFINE_SIMPLE_UNARY_FUNC(Square, int8_t);
DEFINE_SIMPLE_UNARY_FUNC(Square, uint8_t);
DEFINE_SIMPLE_UNARY_FUNC(Square, int);
DEFINE_SIMPLE_UNARY_FUNC(Square, int64_t);
DEFINE_SIMPLE_UNARY_FUNC(Square, float);
DEFINE_SIMPLE_UNARY_FUNC(Square, double);
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

template <typename T>
__global__ void _Set(
    const int               n,
    const T                 alpha,
    T*                      x) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        x[i] = alpha;
    }
}

template <typename T>
__global__ void _Pow(
    const int               n,
    const T                 exp,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = pow(x[i], exp);
    }
}

template <typename T>
__global__ void _Scale(
    const int               n,
    const T                 alpha,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = x[i] * alpha;
    }
}

template <typename T>
__global__ void _Axpy(
    const int               n,
    const T                 alpha,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] += (alpha * x[i]);
    }
}

template <typename T>
__global__ void _Axpby(
    const int               n,
    const T                 alpha,
    const T*                x,
    const T                 beta,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = (alpha * x[i] + beta * y[i]);
    }
}

template <typename T>
__global__ void _AddScalar(
    const int               n,
    const T                 alpha,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] += alpha;
    }
}

/*!                y = a                 */

#define DEFINE_SET_FUNC(T) \
    template <> void Set<T, CUDAContext>( \
        const int               n, \
        const T                 alpha, \
        T*                      y, \
        CUDAContext*            ctx) { \
        if (alpha == (T)0) { \
            CUDA_CHECK(cudaMemsetAsync(y, 0, \
                sizeof(T) * n, ctx->cuda_stream())); \
        } else { \
            _Set<T> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (n, alpha, y); \
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
    template <> void Pow<T, CUDAContext>( \
        const int               n, \
        const float             exp, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Pow<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, exp, x, y); \
    }

DEFINE_POWX_FUNC(float);
DEFINE_POWX_FUNC(double);
#undef DEFINE_POWX_FUNC

/*!        y = ax    ||    x = ax        */

#define DEFINE_SCALE_FUNC(T) \
    template <> void Scale<T, CUDAContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        T _alpha_ = (T)alpha; \
        if (_alpha_ == T(1)) { \
            if (x != y) { \
                cudaMemcpyAsync(y, x, sizeof(T) * n, \
                    cudaMemcpyDeviceToDevice, \
                        ctx->cuda_stream()); \
            } return; \
        } \
        _Scale<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, _alpha_, x, y); \
    }

#define DEFINE_CUBLAS_SCALE_FUNC(T, cublas_func) \
    template <> void Scale<T, CUDAContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        T _alpha_ = (T)alpha; \
        if (x != y) { \
            CUDA_CHECK(cudaMemcpyAsync(y, x, \
                sizeof(T) * n, cudaMemcpyDeviceToDevice, \
                    ctx->cuda_stream())); \
        } \
        if (_alpha_ != T(1)) { \
            CUBLAS_CHECK(cublas_func(  \
                ctx->cublas_handle(), n, &_alpha_, y, 1)); \
        } \
    }

DEFINE_SCALE_FUNC(int8_t);
DEFINE_SCALE_FUNC(uint8_t);
DEFINE_SCALE_FUNC(int);
DEFINE_SCALE_FUNC(int64_t);
#undef DEFINE_SCALE_FUNC
DEFINE_CUBLAS_SCALE_FUNC(float, cublasSscal_v2);
DEFINE_CUBLAS_SCALE_FUNC(double, cublasDscal_v2);
#undef DEFINE_CUBLAS_SCALE_FUNC

/*!                y += ax                */

#define DEFINE_AXPY_FUNC(T) \
    template <> void Axpy<T, CUDAContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Axpy<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, cast::to<T>(alpha), x, y); \
    }

template <> void Axpy<float, CUDAContext>(
    const int               n,
    const float             alpha,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    CUBLAS_CHECK(cublasSaxpy_v2(
        ctx->cublas_handle(), n, &alpha, x, 1, y, 1));
}

template <> void Axpy<double, CUDAContext>(
    const int               n,
    float                   alpha,
    const double*           x,
    double*                 y,
    CUDAContext*            ctx) {
    double alpha64 = alpha;
    CUBLAS_CHECK(cublasDaxpy_v2(
        ctx->cublas_handle(), n, &alpha64, x, 1, y, 1));
}

DEFINE_AXPY_FUNC(int8_t);
DEFINE_AXPY_FUNC(uint8_t);
DEFINE_AXPY_FUNC(int);
DEFINE_AXPY_FUNC(int64_t);
#undef DEFINE_AXPY_FUNC

/*!                 y = ax + by               */

#define DEFINE_AXPBY_FUNC(T) \
    template <> void Axpby<T, CUDAContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        const float             beta, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Axpby<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, cast::to<T>(alpha), x, \
                cast::to<T>(beta), y); \
    }

DEFINE_AXPBY_FUNC(int8_t);
DEFINE_AXPBY_FUNC(uint8_t);
DEFINE_AXPBY_FUNC(int);
DEFINE_AXPBY_FUNC(int64_t);
DEFINE_AXPBY_FUNC(float);
DEFINE_AXPBY_FUNC(double);
#undef DEFINE_AXPBY_FUNC

/*!                 y += a                */

#define DEFINE_ADD_SCALAR_FUNC(T) \
    template <> void AddScalar<T, CUDAContext>( \
        const int               n, \
        const float             alpha, \
        T*                      y, \
        CUDAContext*            ctx) { \
        T _alpha_ = (T)alpha; \
        if (_alpha_ == T(0)) return; \
        _AddScalar<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, _alpha_, y); \
    }

DEFINE_ADD_SCALAR_FUNC(int8_t);
DEFINE_ADD_SCALAR_FUNC(uint8_t);
DEFINE_ADD_SCALAR_FUNC(int);
DEFINE_ADD_SCALAR_FUNC(int64_t);
DEFINE_ADD_SCALAR_FUNC(float);
DEFINE_ADD_SCALAR_FUNC(double);
#undef DEFINE_ADD_SCALAR_FUNC

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

template <typename T>
__global__ void _InvStd(
    const int               n,
    const T                 eps,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        y[i] = rsqrt(x[i] + eps);
    }
}

#define DEFINE_INVSTD_FUNC(T) \
    template <> void InvStd<T, CUDAContext>( \
        const int               n, \
        const float             eps, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _InvStd<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, cast::to<T>(eps), x, y); \
    }

DEFINE_INVSTD_FUNC(float);
DEFINE_INVSTD_FUNC(double);
#undef DEFINE_INVSTD_FUNC

/*!                y = sum(x)               */

#define DEFINE_SUM_FUNC(T) \
    template <> void Sum<T, CUDAContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        vector<int> dims = { n }, axes = { 0 }; \
        kernel::ReduceSum<T, CUDAContext>(1, dims.data(), \
            1, axes.data(), alpha, x, y, ctx); \
    } \
    template <> T Sum<T, CUDAContext>( \
        const int               n, \
        const float             alpha, \
        const T*                x, \
        CUDAContext*            ctx) { \
        T val, *y = (T*)ctx->New(sizeof(T)); \
        Sum<T, CUDAContext>(n, alpha, x, y, ctx); \
        CUDA_CHECK(cudaMemcpyAsync(&val, y, sizeof(T), \
            cudaMemcpyDeviceToHost, ctx->cuda_stream())); \
        ctx->FinishDeviceCompution(); ctx->Delete(y); \
        return val; \
    }

DEFINE_SUM_FUNC(int8_t);
DEFINE_SUM_FUNC(uint8_t);
DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float16);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

/*!                y = sum(abs(x)               */

template <> float ASum<float, CUDAContext>(
    const int               n,
    const float*            x,
    CUDAContext*            ctx) {
    return cublasSasum(n, x, 1);
}

/*!
 * ----------------------------------------------
 *
 *
 *            Simply Binary Functions
 *
 *
 * ----------------------------------------------
 */

#define DEFINE_SIMPLE_BINARY_FUNCTOR(name, expr) \
    template <typename T> \
    __global__ void _##name( \
        const int               n, \
        const T*                a, \
        const T*                b, \
        T*                      y) { \
        CUDA_1D_KERNEL_LOOP(i, n) { \
            y[i] = a[i] expr b[i]; \
        } \
    }

#define DEFINE_SIMPLE_BINARY_FUNC(name, T) \
    template <> void name<T, CUDAContext>( \
        const int               n, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name<T> \
            << < CUDA_BLOCKS(n), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (n, a, b, y); \
    }

DEFINE_SIMPLE_BINARY_FUNCTOR(Add, +);
DEFINE_SIMPLE_BINARY_FUNCTOR(Sub, -);
DEFINE_SIMPLE_BINARY_FUNCTOR(Mul, *);
DEFINE_SIMPLE_BINARY_FUNCTOR(Div, / );
#undef DEFINE_SIMPLE_BINARY_FUNCTOR

DEFINE_SIMPLE_BINARY_FUNC(Add, int8_t);
DEFINE_SIMPLE_BINARY_FUNC(Add, uint8_t);
DEFINE_SIMPLE_BINARY_FUNC(Add, int);
DEFINE_SIMPLE_BINARY_FUNC(Add, int64_t);
DEFINE_SIMPLE_BINARY_FUNC(Add, float);
DEFINE_SIMPLE_BINARY_FUNC(Add, double);
DEFINE_SIMPLE_BINARY_FUNC(Sub, int8_t);
DEFINE_SIMPLE_BINARY_FUNC(Sub, uint8_t);
DEFINE_SIMPLE_BINARY_FUNC(Sub, int);
DEFINE_SIMPLE_BINARY_FUNC(Sub, int64_t);
DEFINE_SIMPLE_BINARY_FUNC(Sub, float);
DEFINE_SIMPLE_BINARY_FUNC(Sub, double);
DEFINE_SIMPLE_BINARY_FUNC(Mul, int8_t);
DEFINE_SIMPLE_BINARY_FUNC(Mul, uint8_t);
DEFINE_SIMPLE_BINARY_FUNC(Mul, int);
DEFINE_SIMPLE_BINARY_FUNC(Mul, int64_t);
DEFINE_SIMPLE_BINARY_FUNC(Mul, float);
DEFINE_SIMPLE_BINARY_FUNC(Mul, double);
DEFINE_SIMPLE_BINARY_FUNC(Div, int8_t);
DEFINE_SIMPLE_BINARY_FUNC(Div, uint8_t);
DEFINE_SIMPLE_BINARY_FUNC(Div, int);
DEFINE_SIMPLE_BINARY_FUNC(Div, int64_t);
DEFINE_SIMPLE_BINARY_FUNC(Div, float);
DEFINE_SIMPLE_BINARY_FUNC(Div, double);
#undef DEFINE_SIMPLE_BINARY_FUNC

#define DEFINE_CUBLAS_DOT_FUNC(T, cublas_func) \
    template <> void Dot<T, CUDAContext>( \
        const int               n, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CUDAContext*            ctx) { \
        CUBLAS_CHECK(cublas_func(ctx->cublas_handle(), \
            n, a, 1, b, 1, y)); \
        ctx->FinishDeviceCompution(); \
    }

DEFINE_CUBLAS_DOT_FUNC(float, cublasSdot_v2);
DEFINE_CUBLAS_DOT_FUNC(double, cublasDdot_v2);
#undef DEFINE_CUBLAS_DOT_FUNC

/*!
 * ----------------------------------------------
 *
 *
 *          Broadcast Binary Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T, bool BroadcastA>
__global__ void _RowBroadcastAdd(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) + __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] + b[b_idx];
#endif
    }
}

template <typename T, bool BroadcastA>
__global__ void _ColBroadcastAdd(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) + __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] + b[b_idx];
#endif
    }
}

template <typename T, bool BroadcastA>
__global__ void _RowBroadcastSub(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) - __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] - b[b_idx];
#endif
    }
}

template <typename T, bool BroadcastA>
__global__ void _ColBroadcastSub(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) - __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] - b[b_idx];
#endif
    }
}

template <typename T, bool BroadcastA>
__global__ void _RowBroadcastMul(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) * __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] * b[b_idx];
#endif
    }
}

template <typename T, bool BroadcastA>
__global__ void _ColBroadcastMul(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) * __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] * b[b_idx];
#endif
    }
}

template <typename T, bool BroadcastA>
__global__ void _RowBroadcastDiv(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) / __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] / b[b_idx];
#endif
    }
}

template <typename T, bool BroadcastA>
__global__ void _ColBroadcastDiv(
    const int               count,
    const int               cols,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(a + a_idx) / __ldg(b + b_idx);
#else
        y[idx] = a[a_idx] / b[b_idx];
#endif
    }
}

#define DEFINE_BROADCAST_BINARY_FUNC(name, T) \
    template <> void Broadcast##name<T, CUDAContext>( \
        const int               rows, \
        const int               cols, \
        const int               type, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto n = rows * cols; \
        if (type == 0) { \
            /*! Row - BroadcastB */ \
            _RowBroadcast##name<T, false> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                    0, ctx->cuda_stream() >> > \
                (n, cols, a, b, y); \
        } else if (type == 1) { \
            /*! Col - BroadcastB */ \
            _ColBroadcast##name<T, false> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                    0, ctx->cuda_stream() >> > \
                (n, cols, a, b, y); \
        } else if (type == 2) { \
            /*! Row - BroadcastA */ \
            _RowBroadcast##name<T, true> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                    0, ctx->cuda_stream() >> > \
                (n, cols, a, b, y); \
        } else if (type == 3) { \
            /*! Col - BroadcastA */ \
            _ColBroadcast##name<T, true> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                    0, ctx->cuda_stream() >> > \
                (n, cols, a, b, y); \
        } else { \
            LOG(FATAL) << "Unknown broadcast type: " << type; \
        } \
    }

DEFINE_BROADCAST_BINARY_FUNC(Add, int8_t);
DEFINE_BROADCAST_BINARY_FUNC(Add, uint8_t);
DEFINE_BROADCAST_BINARY_FUNC(Add, int);
DEFINE_BROADCAST_BINARY_FUNC(Add, int64_t);
DEFINE_BROADCAST_BINARY_FUNC(Add, float);
DEFINE_BROADCAST_BINARY_FUNC(Add, double);
DEFINE_BROADCAST_BINARY_FUNC(Sub, int8_t);
DEFINE_BROADCAST_BINARY_FUNC(Sub, uint8_t);
DEFINE_BROADCAST_BINARY_FUNC(Sub, int);
DEFINE_BROADCAST_BINARY_FUNC(Sub, int64_t);
DEFINE_BROADCAST_BINARY_FUNC(Sub, float);
DEFINE_BROADCAST_BINARY_FUNC(Sub, double);
DEFINE_BROADCAST_BINARY_FUNC(Mul, int8_t);
DEFINE_BROADCAST_BINARY_FUNC(Mul, uint8_t);
DEFINE_BROADCAST_BINARY_FUNC(Mul, int);
DEFINE_BROADCAST_BINARY_FUNC(Mul, int64_t);
DEFINE_BROADCAST_BINARY_FUNC(Mul, float);
DEFINE_BROADCAST_BINARY_FUNC(Mul, double);
DEFINE_BROADCAST_BINARY_FUNC(Div, int8_t);
DEFINE_BROADCAST_BINARY_FUNC(Div, uint8_t);
DEFINE_BROADCAST_BINARY_FUNC(Div, int);
DEFINE_BROADCAST_BINARY_FUNC(Div, int64_t);
DEFINE_BROADCAST_BINARY_FUNC(Div, float);
DEFINE_BROADCAST_BINARY_FUNC(Div, double);
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

template <> void Gemm<float, CUDAContext>(
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
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (TransB == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    const float _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasSgemm_v2(ctx->cublas_handle(),
        cuTransB, cuTransA, N, M, K,
            &_alpha_, B, ldb, A, lda, &_beta_, C, N));
}

template <> void Gemm<double, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const CBLAS_TRANSPOSE   TransB,
    const int               M,
    const int               N,
    const int               K,
    const float             alpha,
    const double*           A,
    const double*           B,
    const float             beta,
    double*                 C,
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (TransB == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    const double _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasDgemm_v2(ctx->cublas_handle(),
        cuTransB, cuTransA, N, M, K,
            &_alpha_, B, ldb, A, lda, &_beta_, C, N));
}

template <> void Gemv<float, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const float*            A,
    const float*            x,
    const float             beta,
    float*                  y,
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_T : CUBLAS_OP_N;
    const float _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasSgemv_v2(
        ctx->cublas_handle(), cuTransA, N, M,
            &_alpha_, A, N, x, 1, &_beta_, y, 1));
}

template <> void Gemv<double, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const double*           A,
    const double*           x,
    const float             beta,
    double*                 y,
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_T : CUBLAS_OP_N;
    const double _alpha_ = alpha, _beta_ = beta;
    CUBLAS_CHECK(cublasDgemv_v2(
        ctx->cublas_handle(), cuTransA, N, M,
            &_alpha_, A, N, x, 1, &_beta_, y, 1));
}

/*!
 * ----------------------------------------------
 *
 *
 *               Random Functions
 *
 *
 * ----------------------------------------------
 */

template <> void RandomUniform<uint32_t, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    uint32_t*               y,
    CUDAContext*            ctx) {
    // Note that we ignore the low / high
    // cuRand could only generates in the range of [0, uint32]
    auto* rng = ctx->curand_generator();
    CURAND_CHECK(curandGenerate(rng, y, n));
}

template <> void RandomUniform<float, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    float*                  y,
    CUDAContext*            ctx) {
    CURAND_CHECK(curandGenerateUniform(
        ctx->curand_generator(), y, n));
    Scale<float, CUDAContext>(n, high - low, y, y, ctx);
    AddScalar<float, CUDAContext>(n, low, y, ctx);
}

template <> void RandomUniform<double, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    double*                 y,
    CUDAContext*            ctx) {
    CURAND_CHECK(curandGenerateUniformDouble(
        ctx->curand_generator(), y, n));
    Scale<double, CUDAContext>(n, high - low, y, y, ctx);
    AddScalar<double, CUDAContext>(n, low, y, ctx);
}

template <> void RandomNormal<float, CUDAContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    float*                  y,
    CUDAContext*            ctx) {
    auto* rng = ctx->curand_generator();
    CURAND_CHECK(curandGenerateNormal(rng, y, n, mu, sigma));
}

template <> void RandomNormal<double, CUDAContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    double*                 y,
    CUDAContext*            ctx) {
    auto* rng = ctx->curand_generator();
    CURAND_CHECK(curandGenerateNormalDouble(rng, y, n, mu, sigma));
}

}  // namespace math

}  // namespace dragon

#endif  // WITH_CUDA