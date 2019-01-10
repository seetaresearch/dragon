  #ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Equal <T = ?, Device = CUDA> */

template <typename T>
__global__ void _EqualInteger(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = a[idx] == b[idx] ? true : false;
    }
}

__global__ void _EqualHalf(
    const int               count,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __heq(a[idx], b[idx]) ? true : false;
#endif
    }
}

template <typename T>
__global__ void _EqualFloat(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = fabs(a[idx] - b[idx]) < 1e-15 ? true : false;
    }
}

/*! Less <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Less(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = a[idx] < b[idx] ? true : false;
    }
}

__global__ void _LessHalf(
    const int               count,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hlt(a[idx], b[idx]) ? true : false;
#endif
    }
}

/*! Greater <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Greater(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = a[idx] > b[idx] ? true : false;
    }
}

__global__ void _GreaterHalf(
    const int               count,
    const half*             a,
    const half*             b,
    bool*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hgt(a[idx], b[idx]) ? true : false;
#endif
    }
}

#define DEFINE_COMPARE_WARPPER(T, OP, IMPL) \
    template <> void OP<T, CUDAContext>( \
        const int               count, \
        const T*                a, \
        const T*                b, \
        bool*                   y, \
        CUDAContext*           ctx) { \
        IMPL<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, a, b, y); \
    }

#define DEFINE_COMPARE_FP16_WARPPER(OP) \
    template <> void OP<float16, CUDAContext>( \
        const int               count, \
        const float16*          a, \
        const float16*          b, \
        bool*                   y, \
        CUDAContext*            ctx) { \
        _##OP##Half \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, reinterpret_cast<const half*>(a), \
                reinterpret_cast<const half*>(b), y); \
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

DEFINE_COMPARE_WARPPER(bool, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int8_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(uint8_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int64_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(float, Greater, _Greater);
DEFINE_COMPARE_WARPPER(double, Greater, _Greater);
DEFINE_COMPARE_FP16_WARPPER(Greater);

#undef DEFINE_COMPARE_WARPPER
#undef DEFINE_COMPARE_FP16_WARPPER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA