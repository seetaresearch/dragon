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
__global__ void _ExpHalf(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = hexp(a[i]);
#endif
    }
}

template <typename T>
__global__ void _ExpHalf2(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = h2exp(a[i]);
#endif
    }
}

template <> void Exp<float16, CUDAContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _ExpHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                    reinterpret_cast<const half2*>(x),
                        reinterpret_cast<half2*>(y));
    }
    else {
        _ExpHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n,
                     reinterpret_cast<const half*>(x),
                        reinterpret_cast<half*>(y));
    }
}

template <typename T>
__global__ void _LogHalf(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = hlog(a[i]);
#endif
    }
}

template <typename T>
__global__ void _LogHalf2(
    const int               n,
    const T*                a,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = h2log(a[i]);
#endif
    }
}

template <> void Log<float16, CUDAContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _LogHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                    reinterpret_cast<const half2*>(x),
                        reinterpret_cast<half2*>(y));
    }
    else {
        _LogHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n,
                     reinterpret_cast<const half*>(x),
                        reinterpret_cast<half*>(y));
    }
}

template <typename T>
__global__ void _InvHalf(
    const int               n,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] =  hrcp(x[i]);
#endif
    }
}

template <typename T>
__global__ void _InvHalf2(
    const int               n,
    const half2*            x,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = h2rcp(x[i]);
#endif
    }
}

template <> void Inv<float16, CUDAContext>(
    const int               n,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _InvHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                     reinterpret_cast<const half2*>(x),
                        reinterpret_cast<half2*>(y));
    } else {
        _InvHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n,
                     reinterpret_cast<const half*>(x),
                        reinterpret_cast<half*>(y));
    }
}

template <typename T>
__global__ void _SqrtHalf(
    int                     n,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = hsqrt(x[i]);
#endif
    }
}

template <typename T>
__global__ void _SqrtHalf2(
    const int               n,
    const half2*            x,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = h2sqrt(x[i]);
#endif
    }
}

template <> void Sqrt<float16, CUDAContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _SqrtHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                     reinterpret_cast<const half2*>(x),
                         reinterpret_cast<half2*>(y));
    } else {
        _SqrtHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n,
                     reinterpret_cast<const half*>(x),
                         reinterpret_cast<half*>(y));
    }
}

template <typename T>
__global__ void _RSqrtHalf(
    int                     n,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = hrsqrt(x[i]);
#endif
    }
}

template <typename T>
__global__ void _RSqrtHalf2(
    const int               n,
    const half2*            x,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = h2rsqrt(x[i]);
#endif
    }
}

template <> void RSqrt<float16, CUDAContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _RSqrtHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                     reinterpret_cast<const half2*>(x),
                         reinterpret_cast<half2*>(y));
    } else {
        _RSqrtHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n,
                     reinterpret_cast<const half*>(x),
                         reinterpret_cast<half*>(y));
    }
}

template <typename T>
__global__ void _SquareHalf(
    const int               n,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul(x[i], x[i]);
#endif
    }
}

template <typename T>
__global__ void _SquareHalf2(
    const int               n,
    const half2*            x,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul2(x[i], x[i]);
#endif
    }
}

template <> void Square<float16, CUDAContext>(
    int                     n,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _SquareHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                     reinterpret_cast<const half2*>(x),
                         reinterpret_cast<half2*>(y));
    } else {
        _SquareHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n,
                     reinterpret_cast<const half*>(x),
                         reinterpret_cast<half*>(y));
    }
}

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

template <typename T>
__global__ void _SetHalf(
    const int               n,
    const T                 alpha,
    T*                      x) {
    CUDA_1D_KERNEL_LOOP(i, n) {
        x[i] = alpha;
    }
}

template <> void Set<float16, CUDAContext>(
    const int               n,
    const float16           alpha,
    float16*                y,
    CUDAContext*            ctx) {
    if (alpha.x == (unsigned short)0) {
        CUDA_CHECK(cudaMemsetAsync(y, 0,
            sizeof(float16) * n, ctx->cuda_stream()));
        return;
    }
    if ((n & 1) == 0) {
        _SetHalf<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                     cast::to<half2>(alpha),
                         reinterpret_cast<half2*>(y));
    } else {
        _SetHalf<float16>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n, alpha, y);
    }
}

/*!                y = x^e                */

template <typename T>
__global__ void _PowHalf(
    const int               n,
    const float             alpha,
    const half*             a,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul(a[i], a[i]);
#endif
    }
}

template <typename T>
__global__ void _PowHalf2(
    const int               n,
    const float             alpha,
    const half2*            a,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul2(a[i], a[i]);
#endif
    }
}

template <> void Pow<float16, CUDAContext>(
    int                     n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    CHECK(alpha == float(2)) << "\nRequired power = 2";
    if ((n & 1) == 0) {
        _PowHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n >> 1,
                     alpha, reinterpret_cast<const half2*>(x),
                         reinterpret_cast<half2*>(y));
    } else {
        _PowHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(n,
                     alpha, reinterpret_cast<const half*>(x),
                         reinterpret_cast<half*>(y));
    }
}

/*!        y = ax    ||    x = ax        */

template <> void Scale<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if (x != y) {
        CUDA_CHECK(cudaMemcpyAsync(y, x, sizeof(float16) * n,
            cudaMemcpyDeviceToDevice, ctx->cuda_stream()));
    }
    if (alpha != 1.f) {
        CUBLAS_CHECK(cublasScalEx(
            ctx->cublas_handle(), n,
                &alpha, CUDA_R_32F,
                    y, CUDA_R_16F, 1,
                        CUDA_R_32F));
    }
}

/*!                y += ax                */

template <> void Axpy<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    CUBLAS_CHECK(cublasAxpyEx(
        ctx->cublas_handle(), n,
            &alpha, CUDA_R_32F,
                x, CUDA_R_16F, 1,
                    y, CUDA_R_16F, 1,
                        CUDA_R_32F));
}

/*!                 y = ax + by               */

template <> void Axpby<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    const float16*          x,
    const float             beta,
    float16*                y,
    CUDAContext*            ctx) {
    Scale(n, beta, y, y, ctx);
    Axpy(n, alpha, x, y, ctx);
}

/*!                 y += a                */

template <typename T>
__global__ void _AddScalarHalf(
    const int               n,
    half                    alpha,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hadd(y[i], alpha);
#endif
    }
}

template <typename T>
__global__ void _AddScalarHalf2(
    const int               n,
    half2                   alpha,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hadd2(y[i], alpha);
#endif
    }
}

template <> void AddScalar<float16, CUDAContext>(
    const int               n,
    const float             alpha,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _AddScalarHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (n >> 1, cast::to<half2>(alpha),
                reinterpret_cast<half2*>(y));
    } else {
        _AddScalarHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (n, cast::to<half>(alpha),
                reinterpret_cast<half*>(y));
    }
}

/*!
 * ----------------------------------------------
 *
 *
 *             Extended Unary Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T>
__global__ void _InvStdHalf(
    int                     n,
    const half              eps,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = hrsqrt(__hadd(x[i], eps));
#endif
    }
}

template <typename T>
__global__ void _InvStdHalf2(
    const int               n,
    const half2             eps,
    const half2*            x,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = h2rsqrt(__hadd2(x[i], eps));
#endif
    }
}

template <> void InvStd<float16, CUDAContext>(
    int                     n,
    const float             eps,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((n & 1) == 0) {
        _InvStdHalf2<half2>
            << < CUDA_BLOCKS(n >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (n >> 1, cast::to<half2>(eps),
                reinterpret_cast<const half2*>(x),
                    reinterpret_cast<half2*>(y));
    } else {
        _InvStdHalf<half>
            << < CUDA_BLOCKS(n), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (n, cast::to<half>(eps),
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    }
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

__global__ void _AddHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hadd(a[i], b[i]);
#endif
    }
}

__global__ void _AddHalf2(
    const int               n,
    const half2*            a,
    const half2*            b,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hadd2(a[i], b[i]);
#endif
    }
}

__global__ void _SubHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hsub(a[i], b[i]);
#endif
    }
}

__global__ void _SubHalf2(
    const int               n,
    const half2*            a,
    const half2*            b,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hsub2(a[i], b[i]);
#endif
    }
}

__global__ void _MulHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul(a[i], b[i]);
#endif
    }
}

__global__ void _MulHalf2(
    const int               n,
    const half2*            a,
    const half2*            b,
    half2*                  y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul2(a[i], b[i]);
#endif
    }
}

__global__ void _DivHalf(
    const int               n,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hdiv(a[i], b[i]);
#endif
    }
}

#define DEFINE_SIMPLE_BINARY_FUNC(name) \
    template <> void name<float16, CUDAContext>( \
        const int               n, \
        const float16*          a, \
        const float16*          b, \
        float16*                y, \
        CUDAContext*            ctx) { \
        if ((n & 1) == 0) { \
            _##name##Half2 \
                << < CUDA_BLOCKS(n >> 1), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (n >> 1, reinterpret_cast<const half2*>(a), \
                    reinterpret_cast<const half2*>(b), \
                        reinterpret_cast<half2*>(y)); \
        } else { \
            _##name##Half \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (n, reinterpret_cast<const half*>(a), \
                    reinterpret_cast<const half*>(b), \
                        reinterpret_cast<half*>(y)); \
        } \
    }

DEFINE_SIMPLE_BINARY_FUNC(Add);
DEFINE_SIMPLE_BINARY_FUNC(Sub);
DEFINE_SIMPLE_BINARY_FUNC(Mul);
#undef DEFINE_SIMPLE_BINARY_FUNC

template <> void Div<float16, CUDAContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CUDAContext*            ctx) {
    _DivHalf
        << < CUDA_BLOCKS(n), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (n, reinterpret_cast<const half*>(a),
            reinterpret_cast<const half*>(b),
                reinterpret_cast<half*>(y));
}

template <> void Dot<float16, CUDAContext>(
    int                     n,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CUDAContext*            ctx) {
    CUBLAS_CHECK(cublasDotEx(
        ctx->cublas_handle(), n,
            a, CUDA_R_16F, 1,
                b, CUDA_R_16F, 1,
                    y, CUDA_R_16F,
                        CUDA_R_32F));
    ctx->FinishDeviceCompution();
}

/*!
 * ----------------------------------------------
 *
 *
 *          Broadcast Binary Functions
 *
 *
 * ----------------------------------------------
 */

template <bool BroadcastA>
__global__ void _RowBroadcastAddHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hadd(a[a_idx], b[b_idx]);
#endif
    }
}

template <bool BroadcastA>
__global__ void _ColBroadcastAddHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hadd(a[a_idx], b[b_idx]);
#endif
    }
}

template <bool BroadcastA>
__global__ void _RowBroadcastSubHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hsub(a[a_idx], b[b_idx]);
#endif
    }
}

template <bool BroadcastA>
__global__ void _ColBroadcastSubHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hsub(a[a_idx], b[b_idx]);
#endif
    }
}

template <bool BroadcastA>
__global__ void _RowBroadcastMulHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hmul(a[a_idx], b[b_idx]);
#endif
    }
}

template <bool BroadcastA>
__global__ void _ColBroadcastMulHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hmul(a[a_idx], b[b_idx]);
#endif
    }
}

template <bool BroadcastA>
__global__ void _RowBroadcastDivHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx % cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hdiv(a[a_idx], b[b_idx]);
#endif
    }
}

template <bool BroadcastA>
__global__ void _ColBroadcastDivHalf(
    const int               count,
    const int               cols,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const int i = idx / cols;
        const int a_idx = BroadcastA ? i : idx;
        const int b_idx = BroadcastA ? idx : i;
        y[idx] = __hdiv(a[a_idx], b[b_idx]);
#endif
    }
}

#define DEFINE_BROADCAST_BINARY_FUNC(name) \
    template <> void Broadcast##name<float16, CUDAContext>( \
        const int               rows, \
        const int               cols, \
        const int               type, \
        const float16*          a, \
        const float16*          b, \
        float16*                y, \
        CUDAContext*            ctx) { \
        auto n = rows * cols; \
        if (type == 0) { \
            /*! Row - BroadcastB */ \
            _RowBroadcast##name##Half<false> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (n, cols, reinterpret_cast<const half*>(a), \
                    reinterpret_cast<const half*>(b), \
                        reinterpret_cast<half*>(y)); \
        } else if (type == 1) { \
            /*! Col - BroadcastB */ \
            _ColBroadcast##name##Half<false> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (n, cols, reinterpret_cast<const half*>(a), \
                    reinterpret_cast<const half*>(b), \
                        reinterpret_cast<half*>(y)); \
        } else if (type == 2) { \
            /*! Row - BroadcastA */ \
            _RowBroadcast##name##Half<true> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (n, cols, reinterpret_cast<const half*>(a), \
                    reinterpret_cast<const half*>(b), \
                        reinterpret_cast<half*>(y)); \
        } else if (type == 3) { \
            /*! Col - BroadcastA */ \
            _ColBroadcast##name##Half<true> \
                << < CUDA_BLOCKS(n), CUDA_THREADS, \
                     0, ctx->cuda_stream() >> > \
                (n, cols, reinterpret_cast<const half*>(a), \
                    reinterpret_cast<const half*>(b), \
                        reinterpret_cast<half*>(y)); \
        } else { \
            LOG(FATAL) << "Unknown broadcast type: " << type; \
        } \
    }

DEFINE_BROADCAST_BINARY_FUNC(Add);
DEFINE_BROADCAST_BINARY_FUNC(Sub);
DEFINE_BROADCAST_BINARY_FUNC(Mul);
DEFINE_BROADCAST_BINARY_FUNC(Div);
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

template <> void Gemm<float16, CUDAContext>(
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
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = (TransB == CblasNoTrans) ?
        CUBLAS_OP_N : CUBLAS_OP_T;
    if (math_type == TensorProto_DataType_FLOAT) {
        const float _alpha_ = alpha, _beta_ = beta;
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMM + MATH32 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    B, CUDA_R_16F, ldb,
                    A, CUDA_R_16F, lda,
                &_beta_,
                    C, CUDA_R_16F, N,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMM + MATH32 + DEFAULT
            CUBLAS_CHECK(cublasSgemmEx(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    B, CUDA_R_16F, ldb,
                    A, CUDA_R_16F, lda,
                &_beta_,
                    C, CUDA_R_16F, N));
        }
#else
       CUBLAS_CHECK(cublasSgemmEx(
           ctx->cublas_handle(),
           cuTransB, cuTransA, N, M, K,
           &_alpha_,
               B, CUDA_R_16F, ldb,
               A, CUDA_R_16F, lda,
           &_beta_,
               C, CUDA_R_16F, N));
#endif
    } else if (math_type == TensorProto_DataType_FLOAT16) {
        const half _alpha_ = cast::to<half>(alpha);
        const half _beta_ = cast::to<half>(beta);
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMM + MATH16 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    B, CUDA_R_16F, ldb,
                    A, CUDA_R_16F, lda,
                &_beta_,
                    C, CUDA_R_16F, N,
                CUDA_R_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMM + MATH16 + DEFAULT
            CUBLAS_CHECK(cublasHgemm(
                ctx->cublas_handle(),
                cuTransB, cuTransA, N, M, K,
                &_alpha_,
                    reinterpret_cast<const half*>(B), ldb,
                    reinterpret_cast<const half*>(A), lda,
                &_beta_,
                    reinterpret_cast<half*>(C), N));
        }
#else
        CUBLAS_CHECK(cublasHgemm(
            ctx->cublas_handle(),
            cuTransB, cuTransA, N, M, K,
            &_alpha_,
                reinterpret_cast<const half*>(B), ldb,
                reinterpret_cast<const half*>(A), lda,
            &_beta_,
                reinterpret_cast<half*>(C), N));
#endif
    } else {
        LOG(FATAL) << "Unsupported math type";
    }
}

template <> void Gemv<float16, CUDAContext>(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const float16*          A,
    const float16*          x,
    const float             beta,
    float16*                y,
    CUDAContext*            ctx,
    TensorProto_DataType    math_type) {
    cublasOperation_t cuTransA = (TransA == CblasNoTrans) ?
        CUBLAS_OP_T : CUBLAS_OP_N;
    int m = (cuTransA == CUBLAS_OP_N) ? N : M;
    int k = (cuTransA == CUBLAS_OP_N) ? M : N;
    int LDA = (cuTransA == CUBLAS_OP_N) ? m : k;
    int LDC = m;
    const float _alpha_ = alpha, _beta_ = beta;
    if (math_type == TensorProto_DataType_FLOAT) {
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMV + MATH32 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    A, CUDA_R_16F, LDA,
                    x, CUDA_R_16F, k,
                &_beta_,
                    y, CUDA_R_16F, LDC,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMV + MATH32 + DEFAULT
            CUBLAS_CHECK(cublasSgemmEx(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    A, CUDA_R_16F, LDA,
                    x, CUDA_R_16F, k,
                &_beta_,
                    y, CUDA_R_16F, LDC));
        }
#else
        CUBLAS_CHECK(cublasSgemmEx(
            ctx->cublas_handle(),
            cuTransA, CUBLAS_OP_N, m, 1, k,
            &_alpha_,
                A, CUDA_R_16F, LDA,
                x, CUDA_R_16F, k,
            &_beta_,
                y, CUDA_R_16F, LDC));
#endif
    } else if (math_type == TensorProto_DataType_FLOAT16) {
        const half _alpha_ = cast::to<half>(alpha);
        const half _beta_ = cast::to<half>(beta);
#if CUDA_VERSION >= 9000
        if (TENSOR_CORE_AVAILABLE()) {
            //  GEMV + MATH16 + TENSOR-CORE
            CUBLAS_CHECK(cublasGemmEx(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    A, CUDA_R_16F, LDA,
                    x, CUDA_R_16F, k,
                &_beta_,
                    y, CUDA_R_16F, LDC,
                CUDA_R_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            //  GEMV + MATH16 + DEFAULT
            CUBLAS_CHECK(cublasHgemm(
                ctx->cublas_handle(),
                cuTransA, CUBLAS_OP_N, m, 1, k,
                &_alpha_,
                    reinterpret_cast<const half*>(A), LDA,
                    reinterpret_cast<const half*>(x), k,
                &_beta_,
                    reinterpret_cast<half*>(y), LDC));
        }
#else
        CUBLAS_CHECK(cublasHgemm(
            ctx->cublas_handle(),
            cuTransA, CUBLAS_OP_N, m, 1, k,
            &_alpha_,
                reinterpret_cast<const half*>(A), LDA,
                reinterpret_cast<const half*>(x), k,
            &_beta_,
                reinterpret_cast<half*>(y), LDC));
#endif
    } else {
            LOG(FATAL) << "Unsupported math type";
    }
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

template <> void RandomUniform<float16, CUDAContext>(
    const int               n,
    const float             low,
    const float             high,
    float16*                y,
    CUDAContext*            ctx) {
    auto* y32 = (float*)ctx->New(n * sizeof(float));
    math::RandomUniform<float, CUDAContext>(n, low, high, y32, ctx);
    kernel::TypeA2B<float, float16>(n, y32, y, ctx);
    ctx->FinishDeviceCompution();  // Sync the stream
    ctx->Delete(y32);
}

template <> void RandomNormal<float16, CUDAContext>(
    const int               n,
    const float             mu,
    const float             sigma,
    float16*                y,
    CUDAContext*            ctx) {
    auto* y32 = (float*)ctx->New(n * sizeof(float));
    math::RandomNormal<float, CUDAContext>(n, mu, sigma, y32, ctx);
    kernel::TypeA2B<float, float16>(n, y32, y, ctx);
    ctx->FinishDeviceCompution();  // Sync the stream
    ctx->Delete(y32);
}

}  // namespace math

}  // namespace dragon

#endif  // WITH_CUDA