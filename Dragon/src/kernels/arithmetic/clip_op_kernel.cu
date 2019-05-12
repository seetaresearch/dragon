#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Clip(
    const int               nthreads,
    const T                 low,
    const T                 high,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = max(low, min(x[i], high));
    }
}

template<> __global__ void _Clip<half>(
    const int               nthreads,
    const half              low,
    const half              high,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const half xi = __hlt(x[i], high) ? x[i] : high;
        y[i] = __hgt(xi, low) ? xi : low;
#endif
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _ClipGrad(
    const int               nthreads,
    const T                 low,
    const T                 high,
    const T*                x,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const T xi = x[i];
        dx[i] = (xi < low || xi > high) ? T(0) : dy[i];
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _ClipGrad<half>(
    const int               nthreads,
    const half              low,
    const half              high,
    const half*             x,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const half xi = x[i];
        bool is_zero = false;
        is_zero |= __hlt(xi, low);
        is_zero |= __hgt(xi, high);
        dx[i] = is_zero ? __float2half(0.f) : dy[i];
#endif
    }
}

/* Kernel Launchers */

#define DEFINE_CLIP_KERNEL_LAUNCHER(T) \
    template <> void Clip<T, CUDAContext>( \
        const int               count, \
        const float             low, \
        const float             high, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Clip<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count,  \
            cast::to<T>(low), \
            cast::to<T>(high), \
            x, y \
        ); \
    }

#define DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(T) \
    template <> void ClipGrad<T, CUDAContext>( \
        const int               count, \
        const float             low, \
        const float             high, \
        const T*                x, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        _ClipGrad<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, \
            cast::to<T>(low), \
            cast::to<T>(high), \
            x, dy, dx \
        ); \
    }

DEFINE_CLIP_KERNEL_LAUNCHER(int8_t);
DEFINE_CLIP_KERNEL_LAUNCHER(uint8_t);
DEFINE_CLIP_KERNEL_LAUNCHER(int);
DEFINE_CLIP_KERNEL_LAUNCHER(int64_t);
DEFINE_CLIP_KERNEL_LAUNCHER(float);
DEFINE_CLIP_KERNEL_LAUNCHER(double);

DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(int);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(float);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(double);

template <> void Clip<float16, CUDAContext>(
    const int               count,
    const float             low,
    const float             high,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _Clip
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        cast::to<half>(low),
        cast::to<half>(high),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y)
    );
}

template <> void ClipGrad<float16, CUDAContext>(
    const int               count,
    const float             low,
    const float             high,
    const float16*          x,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    _ClipGrad
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        cast::to<half>(low),
        cast::to<half>(high),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx)
    );
}

#undef DEFINE_CLIP_KERNEL_LAUNCHER
#undef DEFINE_CLIP_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA