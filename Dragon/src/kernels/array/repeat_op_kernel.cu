#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Repeat(
    const int               nthreads,
    const int               axis_dim,
    const int               x_inner_dim,
    const int               y_inner_dim,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int inner_idx = yi % x_inner_dim;
        const int rep_idx = (yi / y_inner_dim) % axis_dim;
        const int outer_idx = yi / y_inner_dim / axis_dim;
        const int xi = (
            outer_idx * axis_dim + rep_idx
                ) * x_inner_dim + inner_idx;
        y[yi] = x[xi];
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _RepeatGrad(
    const int               nthreads,
    const int               axis_dim,
    const int               x_inner_dim,
    const int               y_inner_dim,
    const int               repeats,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        const int inner_idx = xi % x_inner_dim;
        const int rep_idx = (xi / x_inner_dim) % axis_dim;
        const int outer_idx = xi / x_inner_dim / axis_dim;
        const T* dY = dy + (
            (outer_idx * axis_dim + rep_idx
                ) * y_inner_dim + inner_idx);
        T gradient = 0;
        for (int r = 0; r < repeats; ++r)
            gradient += dY[r * x_inner_dim];
        dx[xi] = gradient;
    }
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _RepeatGrad<half>(
    const int               nthreads,
    const int               axis_dim,
    const int               x_inner_dim,
    const int               y_inner_dim,
    const int               repeats,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int inner_idx = xi % x_inner_dim;
        const int rep_idx = (xi / x_inner_dim) % axis_dim;
        const int outer_idx = xi / x_inner_dim / axis_dim;
        const half* dY = dy + (
            (outer_idx * axis_dim + rep_idx
                ) * y_inner_dim + inner_idx);
        float gradient = 0.f;
        for (int r = 0; r < repeats; ++r)
            gradient += __half2float(dY[r * x_inner_dim]);
        dx[xi] = __float2half(gradient);
#endif
    }
}

/* Kernel Launchers */

#define DEFINE_REPEAT_KERNEL_LAUNCHER(T) \
    template<> void Repeat<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               repeats, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto y_inner_dim = inner_dim * repeats; \
        auto nthreads = outer_dim * axis_dim * y_inner_dim; \
        _Repeat \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            nthreads, axis_dim, \
            inner_dim, y_inner_dim, \
            x, y \
        ); \
    }

#define DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(T) \
    template<> void RepeatGrad<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               repeats, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        auto y_inner_dim = inner_dim * repeats; \
        auto nthreads = outer_dim * axis_dim * inner_dim; \
        _RepeatGrad \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            nthreads, \
            axis_dim, \
            inner_dim, \
            y_inner_dim, \
            repeats, \
            dy, dx \
        ); \
    }

DEFINE_REPEAT_KERNEL_LAUNCHER(bool);
DEFINE_REPEAT_KERNEL_LAUNCHER(int8_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(uint8_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(int);
DEFINE_REPEAT_KERNEL_LAUNCHER(int64_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(float16);
DEFINE_REPEAT_KERNEL_LAUNCHER(float);
DEFINE_REPEAT_KERNEL_LAUNCHER(double);

DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(int);
DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(float);
DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(double);

template<> void RepeatGrad<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               repeats,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    auto y_inner_dim = inner_dim * repeats;
    auto nthreads = outer_dim * axis_dim * inner_dim;
    _RepeatGrad
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
             0, ctx->cuda_stream() >> >(
        nthreads,
        axis_dim,
        inner_dim,
        y_inner_dim,
        repeats,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx)
    );
}

#undef DEFINE_REPEAT_KERNEL_LAUNCHER
#undef DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA