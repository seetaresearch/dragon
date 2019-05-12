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
    const T*                x1,
    const T*                x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = max(x1[i], x2[i]);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _Maximum<half>(
    const int               nthreads,
    const half*             x1,
    const half*             x2,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hgt(x1[i], x2[i]) ? x1[i] : x2[i];
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMaximum(
    const int               nthreads,
    const T*                x1,
    const T                 x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = max(x1[i], x2);
    }
}

/* <T = float16, Device = CUDA> */

template<>  __global__ void _BroadcastMaximum<half>(
    const int               nthreads,
    const half*             x1,
    const half              x2,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hgt(x1[i], x2) ? x1[i] : x2;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _MaximumGrad(
    const int               nthreads,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const bool dy_to_dx1 = x1[i] > x2[i];
        dx1[i] = dy_to_dx1 ? dy[i] : T(0);
        dx2[i] = dy_to_dx1 ? T(0) : dy[i];
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _MaximumGrad<half>(
    const int               nthreads,
    const half*             x1,
    const half*             x2,
    const half*             dy,
    half*                   dx1,
    half*                   dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const bool dy_to_dx1 = __hgt(x1[i], x2[i]);
        dx1[i] = dy_to_dx1 ? dy[i] : __float2half(0.f);
        dx2[i] = dy_to_dx1 ? __float2half(0.f) : dy[i];
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMaximumGrad(
    const int               nthreads,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        dx1[i] = (x1[i] > x2) ? dy[i] : T(0);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _BroadcastMaximumGrad<half>(
    const int               nthreads,
    const half*             x1,
    const half              x2,
    const half*             dy,
    half*                   dx1,
    half*                   dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        dx1[i] = __hgt(x1[i], x2) ?
            dy[i] : __float2half(0.f);
#endif
    }
}

/* Kernel Launchers */

#define DEFINE_MAXIMUM_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CUDAContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, x1, x2, y \
        ); \
    }

#define DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CUDAContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        const T*                dy, \
        T*                      dx1, \
        T*                      dx2, \
        CUDAContext*            ctx) { \
        _##name \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, x1, x2, dy, dx1, dx2 \
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
    const float16*          x1,
    const float16*          x2,
    float16*                y,
    CUDAContext*            ctx) {
    _Maximum \
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        reinterpret_cast<const half*>(x1),
        reinterpret_cast<const half*>(x2),
        reinterpret_cast<half*>(y)
    );
}

template <> void BroadcastMaximum<float16, CUDAContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    float16*                y,
    CUDAContext*            ctx) {
    _BroadcastMaximum \
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        reinterpret_cast<const half*>(x1),
        cast::to<half>(x2),
        reinterpret_cast<half*>(y)
    );
}

template <> void MaximumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          x1,
    const float16*          x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CUDAContext*            ctx) {
    _MaximumGrad \
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        reinterpret_cast<const half*>(x1),
        reinterpret_cast<const half*>(x2),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx1),
        reinterpret_cast<half*>(dx2)
    );
}

template <> void BroadcastMaximumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CUDAContext*            ctx) {
    _BroadcastMaximumGrad \
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        reinterpret_cast<const half*>(x1),
        cast::to<half>(x2),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx1),
        reinterpret_cast<half*>(dx2)
    );
}

#undef DEFINE_MAXIMUM_KERNEL_LAUNCHER
#undef DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA