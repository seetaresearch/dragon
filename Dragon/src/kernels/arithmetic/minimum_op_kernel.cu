#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Minimum(
    const int               nthreads,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = min(a[i], b[i]);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _Minimum<half>(
    const int               nthreads,
    const half*             a,
    const half*             b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hlt(a[i], b[i]) ? a[i] : b[i];
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMinimum(
    const int               nthreads,
    const T*                a,
    const T                 b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = min(a[i], b);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _BroadcastMinimum<half>(
    const int               nthreads,
    const half*             a,
    const half              b,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hlt(a[i], b) ? a[i] : b;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _MinimumGrad(
    const int               nthreads,
    const T*                a,
    const T*                b,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const bool dy_to_da = a[i] < b[i];
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _MinimumGrad<half>(
    const int               nthreads,
    const half*             a,
    const half*             b,
    const half*             dy,
    half*                   da,
    half*                   db) {
#if __CUDA_ARCH__ >= 530
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const bool dy_to_da = __hlt(a[i], b[i]);
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
#endif
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMinimumGrad(
    const int               nthreads,
    const T*                a,
    const T                 b,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        da[i] = (a[i] < b) ? dy[i] : kZero;
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _BroadcastMinimumGrad<half>(
    const int               nthreads,
    const half*             a,
    const half              b,
    const half*             dy,
    half*                   da,
    half*                   db) {
#if __CUDA_ARCH__ >= 530
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        da[i] = __hlt(a[i], b) ? dy[i] : kZero;
    }
#endif
}

/* Kernel Launchers */

#define DEFINE_MINIMUM_KERNEL_LAUNCHER(name, T, T2) \
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

#define DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
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

DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, int8_t, int8_t*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, uint8_t, uint8_t*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, int, int*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, int64_t, int64_t*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, float, float*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, double, double*);

DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, int8_t, int8_t);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, uint8_t, uint8_t);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, int, int);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, int64_t, int64_t);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, float, float);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, double, double);

DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, int8_t, int8_t*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, uint8_t, uint8_t*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, int, int*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, int64_t, int64_t*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, float, float*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, double, double*);

DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, int8_t, int8_t);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, uint8_t, uint8_t);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, int, int);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, int64_t, int64_t);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, float, float);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, double, double);

template <> void Minimum<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CUDAContext*            ctx) {
    _Minimum \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(a),
        reinterpret_cast<const half*>(b),
        reinterpret_cast<half*>(y)
    );
}

template <> void BroadcastMinimum<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    float16*                y,
    CUDAContext*            ctx) {
    _BroadcastMinimum \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(a),
        cast::to<half>(b),
        reinterpret_cast<half*>(y)
    );
}

template <> void MinimumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CUDAContext*            ctx) {
    _MinimumGrad \
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

template <> void BroadcastMinimumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CUDAContext*            ctx) {
    _BroadcastMinimumGrad \
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

#undef DEFINE_MINIMUM_KERNEL_LAUNCHER
#undef DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA