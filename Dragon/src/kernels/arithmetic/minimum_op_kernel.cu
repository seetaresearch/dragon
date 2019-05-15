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
    const T*                x1,
    const T*                x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = min(x1[i], x2[i]);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _Minimum<half>(
    const int               nthreads,
    const half*             x1,
    const half*             x2,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hlt(x1[i], x2[i]) ? x1[i] : x2[i];
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMinimum(
    const int               nthreads,
    const T*                x1,
    const T                 x2,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = min(x1[i], x2);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _BroadcastMinimum<half>(
    const int               nthreads,
    const half*             x1,
    const half              x2,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hlt(x1[i], x2) ? x1[i] : x2;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _MinimumGrad(
    const int               nthreads,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const bool dy_to_dx1 = x1[i] < x2[i];
        dx1[i] = dy_to_dx1 ? dy[i] : T(0);
        dx2[i] = dy_to_dx1 ? T(0) : dy[i];
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _MinimumGrad<half>(
    const int               nthreads,
    const half*             x1,
    const half*             x2,
    const half*             dy,
    half*                   dx1,
    half*                   dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const bool dy_to_dx1 = __hlt(x1[i], x2[i]);
        dx1[i] = dy_to_dx1 ? dy[i] : __float2half(0.f);
        dx2[i] = dy_to_dx1 ? __float2half(0.f) : dy[i];
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _BroadcastMinimumGrad(
    const int               nthreads,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        dx1[i] = (x1[i] < x2) ? dy[i] : T(0);
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _BroadcastMinimumGrad<half>(
    const int               nthreads,
    const half*             x1,
    const half              x2,
    const half*             dy,
    half*                   dx1,
    half*                   dx2) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        dx1[i] = (__hlt(x1[i], x2)) ?
            dy[i] : __float2half(0.f);
#endif
    }
}

/* Kernel Launchers */

#define DEFINE_MINIMUM_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CUDAContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, x1, x2, y \
        ); \
    }

#define DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CUDAContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        const T*                dy, \
        T*                      dx1, \
        T*                      dx2, \
        CUDAContext*            ctx) { \
        _##name \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, x1, x2, dy, dx1, dx2 \
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
    const float16*          x1,
    const float16*          x2,
    float16*                y,
    CUDAContext*            ctx) {
    _Minimum \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(x1),
        reinterpret_cast<const half*>(x2),
        reinterpret_cast<half*>(y)
    );
}

template <> void BroadcastMinimum<float16, CUDAContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    float16*                y,
    CUDAContext*            ctx) {
    _BroadcastMinimum \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(x1),
        cast::to<half>(x2),
        reinterpret_cast<half*>(y)
    );
}

template <> void MinimumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          x1,
    const float16*          x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CUDAContext*            ctx) {
    _MinimumGrad \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(x1),
        reinterpret_cast<const half*>(x2),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx1),
        reinterpret_cast<half*>(dx2)
    );
}

template <> void BroadcastMinimumGrad<float16, CUDAContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CUDAContext*            ctx) {
    _BroadcastMinimumGrad \
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(x1),
        cast::to<half>(x2),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx1),
        reinterpret_cast<half*>(dx2)
    );
}

#undef DEFINE_MINIMUM_KERNEL_LAUNCHER
#undef DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA