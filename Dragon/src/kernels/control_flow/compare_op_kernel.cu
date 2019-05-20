  #ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _NotZero(
    const int               nthreads,
    const T*                x,
    bool*                   y) {
    const T kZero = T(0);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = x[i] != kZero ? true : false;
    }
}

template<> __global__ void _NotZero<half>(
    const int               nthreads,
    const half*             x,
    bool*                   y) {
#if __CUDA_ARCH__ >= 530
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = __hne(x[i], kZero) ? true : false;
    }
#endif
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _EqualInteger(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = a[i] == b[i] ? true : false;
    }
}

__global__ void _EqualHalf(
    const int               nthreads,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __heq(a[i], b[i]) ? true : false;
#endif
    }
}

template <typename T>
__global__ void _EqualFloat(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = fabs(a[i] - b[i]) < 1e-15 ? true : false;
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _NotEqualInteger(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = a[i] != b[i] ? true : false;
    }
}

__global__ void _NotEqualHalf(
    const int               nthreads,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hne(a[i], b[i]) ? true : false;
#endif
    }
}

template <typename T>
__global__ void _NotEqualFloat(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = fabs(a[i] - b[i]) > 1e-15 ? true : false;
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Less(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = a[i] < b[i] ? true : false;
    }
}

__global__ void _LessHalf(
    const int               nthreads,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hlt(a[i], b[i]) ? true : false;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _LessEqual(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = a[i] <= b[i] ? true : false;
    }
}

__global__ void _LessEqualHalf(
    const int               nthreads,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hle(a[i], b[i]) ? true : false;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Greater(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = a[i] > b[i] ? true : false;
    }
}

__global__ void _GreaterHalf(
    const int               nthreads,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hgt(a[i], b[i]) ? true : false;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _GreaterEqual(
    const int               nthreads,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = a[i] >= b[i] ? true : false;
    }
}

__global__ void _GreaterEqualHalf(
    const int               nthreads,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hge(a[i], b[i]) ? true : false;
#endif
    }
}

/* Kernel Launchers */

#define DEFINE_NOTZERO_KERNEL_LAUNCHER(T) \
    template <> void NotZero<T, CUDAContext>( \
        const int               count, \
        const T*                x, \
        bool*                   y, \
        CUDAContext*            ctx) { \
        _NotZero \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, x, y \
        ); \
    }

#define DEFINE_COMPARE_KERNEL_LAUNCHER(T, OP, IMPL) \
    template <> void OP<T, CUDAContext>( \
        const int               count, \
        const T*                a, \
        const T*                b, \
        bool*                   y, \
        CUDAContext*            ctx) { \
        IMPL \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, a, b, y \
        ); \
    }

#define DEFINE_COMPARE_FP16_KERNEL_LAUNCHER(OP) \
    template <> void OP<float16, CUDAContext>( \
        const int               count, \
        const float16*          a, \
        const float16*          b, \
        bool*                   y, \
        CUDAContext*            ctx) { \
        _##OP##Half \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, \
            reinterpret_cast<const half*>(a), \
            reinterpret_cast<const half*>(b), \
            y \
        ); \
    }

DEFINE_NOTZERO_KERNEL_LAUNCHER(bool);
DEFINE_NOTZERO_KERNEL_LAUNCHER(int8_t);
DEFINE_NOTZERO_KERNEL_LAUNCHER(uint8_t);
DEFINE_NOTZERO_KERNEL_LAUNCHER(int);
DEFINE_NOTZERO_KERNEL_LAUNCHER(int64_t);
DEFINE_NOTZERO_KERNEL_LAUNCHER(float);
DEFINE_NOTZERO_KERNEL_LAUNCHER(double);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, Equal, _EqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, Equal, _EqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, Equal, _EqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, Equal, _EqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, Equal, _EqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, Equal, _EqualFloat);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, Equal, _EqualFloat);
DEFINE_COMPARE_FP16_KERNEL_LAUNCHER(Equal);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, NotEqual, _NotEqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, NotEqual, _NotEqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, NotEqual, _NotEqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, NotEqual, _NotEqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, NotEqual, _NotEqualInteger);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, NotEqual, _NotEqualFloat);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, NotEqual, _NotEqualFloat);
DEFINE_COMPARE_FP16_KERNEL_LAUNCHER(NotEqual);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, Less, _Less);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, Less, _Less);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, Less, _Less);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, Less, _Less);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, Less, _Less);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, Less, _Less);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, Less, _Less);
DEFINE_COMPARE_FP16_KERNEL_LAUNCHER(Less);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, LessEqual, _LessEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, LessEqual, _LessEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, LessEqual, _LessEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, LessEqual, _LessEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, LessEqual, _LessEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, LessEqual, _LessEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, LessEqual, _LessEqual);
DEFINE_COMPARE_FP16_KERNEL_LAUNCHER(LessEqual);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, Greater, _Greater);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, Greater, _Greater);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, Greater, _Greater);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, Greater, _Greater);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, Greater, _Greater);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, Greater, _Greater);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, Greater, _Greater);
DEFINE_COMPARE_FP16_KERNEL_LAUNCHER(Greater);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_FP16_KERNEL_LAUNCHER(GreaterEqual);

template <> void NotZero<float16, CUDAContext>(
    const int               count,
    const float16*          x,
    bool*                   y,
    CUDAContext*            ctx) {
    _NotZero
        <<< CUDA_BLOCKS(count), CUDA_THREADS, \
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(x),
        y
    );
}

#undef DEFINE_NOTZERO_KERNEL_LAUNCHER
#undef DEFINE_COMPARE_KERNEL_LAUNCHER
#undef DEFINE_COMPARE_FP16_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA