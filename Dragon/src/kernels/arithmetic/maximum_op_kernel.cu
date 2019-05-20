#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Maximum(
    const int               nthreads,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = max(a[i], b[i]);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _Maximum<half>(
    const int               nthreads,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hgt(a[i], b[i]) ? a[i] : b[i];
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMaximum(
    const int               nthreads,
    const T*                a,
    const T                 b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = max(a[i], b);
    }
}

/* <T = float16, Device = CUDA> */

template<>  __global__ void _BroadcastMaximum<half>(
    const int               nthreads,
    const half*             a,
    const half              b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hgt(a[i], b) ? a[i] : b;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _MaximumGrad(
    const int               nthreads,
    const T*                a,
    const T*                b,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const bool dy_to_da = a[i] > b[i];
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _MaximumGrad<half>(
    const int               nthreads,
    const half*             a,
    const half*             b,
    const half*             dy,
    half*                   da,
    half*                   db) {
#if __CUDA_ARCH__ >= 530
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const bool dy_to_da = __hgt(a[i], b[i]);
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
#endif
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMaximumGrad(
    const int               nthreads,
    const T*                a,
    const T                 b,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        da[i] = (a[i] > b) ? dy[i] : kZero;
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _BroadcastMaximumGrad<half>(
    const int               nthreads,
    const half*             a,
    const half              b,
    const half*             dy,
    half*                   da,
    half*                   db) {
#if __CUDA_ARCH__ >= 530
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        da[i] = __hgt(a[i], b) ? dy[i] : kZero;
    }
#endif
}

/* Kernel Launchers */

#define DEFINE_MAXIMUM_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CUDAContext>( \
        const int               count, \
        const T*                a, \
        const T2                b, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, a, b, y \
        ); \
    }

#define DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CUDAContext>( \
        const int               count, \
        const T*                a, \
        const T2                b, \
        const T*                dy, \
        T*                      da, \
        T*                      db, \
        CUDAContext*            ctx) { \
        _##name \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, a, b, dy, da, db \
        ); \
    }

DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, int8_t, int8_t*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, uint8_t, uint8_t*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, int, int*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, int64_t, int64_t*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, float, float*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, double, double*);

DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, int8_t, int8_t);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, uint8_t, uint8_t);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, int, int);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, int64_t, int64_t);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, float, float);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, double, double);

DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, int8_t, int8_t*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, uint8_t, uint8_t*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, int, int*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, int64_t, int64_t*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, float, float*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, double, double*);

DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, int8_t, int8_t);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, uint8_t, uint8_t);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, int, int);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, int64_t, int64_t);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, float, float);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, double, double);

template <> void Maximum<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CUDAContext*            ctx) {
    _Maximum \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(a),
        reinterpret_cast<const half*>(b),
        reinterpret_cast<half*>(y)
    );
}

template <> void BroadcastMaximum<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    float16*                y,
    CUDAContext*            ctx) {
    _BroadcastMaximum \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(a),
        cast::to<half>(b),
        reinterpret_cast<half*>(y)
    );
}

template <> void MaximumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CUDAContext*            ctx) {
    _MaximumGrad \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(a),
        reinterpret_cast<const half*>(b),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(da),
        reinterpret_cast<half*>(db)
    );
}

template <> void BroadcastMaximumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CUDAContext*            ctx) {
    _BroadcastMaximumGrad \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(a),
        cast::to<half>(b),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(da),
        reinterpret_cast<half*>(db)
    );
}

#undef DEFINE_MAXIMUM_KERNEL_LAUNCHER
#undef DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA