  #ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

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

#define DEFINE_COMPARE_WARPPER(T, OP, IMPL) \
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

#define DEFINE_COMPARE_FP16_WARPPER(OP) \
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

DEFINE_COMPARE_WARPPER(bool, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(int8_t, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(uint8_t, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(int, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(int64_t, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(float, Equal, _EqualFloat);
DEFINE_COMPARE_WARPPER(double, Equal, _EqualFloat);
DEFINE_COMPARE_FP16_WARPPER(Equal);

DEFINE_COMPARE_WARPPER(bool, Less, _Less);
DEFINE_COMPARE_WARPPER(int8_t, Less, _Less);
DEFINE_COMPARE_WARPPER(uint8_t, Less, _Less);
DEFINE_COMPARE_WARPPER(int, Less, _Less);
DEFINE_COMPARE_WARPPER(int64_t, Less, _Less);
DEFINE_COMPARE_WARPPER(float, Less, _Less);
DEFINE_COMPARE_WARPPER(double, Less, _Less);
DEFINE_COMPARE_FP16_WARPPER(Less);

DEFINE_COMPARE_WARPPER(bool, LessEqual, _LessEqual);
DEFINE_COMPARE_WARPPER(int8_t, LessEqual, _LessEqual);
DEFINE_COMPARE_WARPPER(uint8_t, LessEqual, _LessEqual);
DEFINE_COMPARE_WARPPER(int, LessEqual, _LessEqual);
DEFINE_COMPARE_WARPPER(int64_t, LessEqual, _LessEqual);
DEFINE_COMPARE_WARPPER(float, LessEqual, _LessEqual);
DEFINE_COMPARE_WARPPER(double, LessEqual, _LessEqual);
DEFINE_COMPARE_FP16_WARPPER(LessEqual);

DEFINE_COMPARE_WARPPER(bool, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int8_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(uint8_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int64_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(float, Greater, _Greater);
DEFINE_COMPARE_WARPPER(double, Greater, _Greater);
DEFINE_COMPARE_FP16_WARPPER(Greater);

DEFINE_COMPARE_WARPPER(bool, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_WARPPER(int8_t, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_WARPPER(uint8_t, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_WARPPER(int, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_WARPPER(int64_t, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_WARPPER(float, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_WARPPER(double, GreaterEqual, _GreaterEqual);
DEFINE_COMPARE_FP16_WARPPER(GreaterEqual);

#undef DEFINE_COMPARE_WARPPER
#undef DEFINE_COMPARE_FP16_WARPPER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA