#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Repeat <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Repeat(
    const int               nthreads,
    const int               repeat_dim,
    const int               x_inner_dim,
    const int               y_inner_dim,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int iix = y_idx % x_inner_dim;
        const int rix = (y_idx / y_inner_dim) % repeat_dim;
        const int oix = y_idx / y_inner_dim / repeat_dim;
        const int x_idx = (oix * repeat_dim + rix)
                            * x_inner_dim + iix;
        y[y_idx] = x[x_idx];
    }
}

/*! RepeatGrad <T = ?, Device = CUDA> */

template <typename T>
__global__ void _RepeatGrad(
    const int               nthreads,
    const int               repeat_dim,
    const int               x_inner_dim,
    const int               y_inner_dim,
    const int               repeats,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        const int iix = x_idx % x_inner_dim;
        const int rix = (x_idx / x_inner_dim) % repeat_dim;
        const int oix = x_idx / x_inner_dim  / repeat_dim;
        const T* dY = dy + ((oix * repeat_dim + rix)
                              * y_inner_dim + iix);

        T gradient = 0;
        for (int r = 0; r < repeats; ++r)
            gradient += dY[r * x_inner_dim];

        dx[x_idx] = gradient;
    }
}

/*! RepeatGrad <T = float16, Device = CUDA> */

__global__ void _RepeatGradHalf(
    const int               nthreads,
    const int               repeat_dim,
    const int               x_inner_dim,
    const int               y_inner_dim,
    const int               repeats,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int iix = x_idx % x_inner_dim;
        const int rix = (x_idx / x_inner_dim) % repeat_dim;
        const int oix = x_idx / x_inner_dim / repeat_dim;
        const half* dY = dy + ((oix * repeat_dim + rix)
            * y_inner_dim + iix);

        float gradient = 0;
        for (int r = 0; r < repeats; ++r)
            gradient += __half2float(dY[r * x_inner_dim]);

        dx[x_idx] = __float2half(gradient);
#endif
    }
}

/*! Kernel Launchers */

#define DEFINE_REPEAT_KERNEL_LAUNCHER(T) \
    template<> void Repeat<T, CUDAContext>( \
        const int               outer_dim, \
        const int               repeat_dim, \
        const int               inner_dim, \
        const int               repeats, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto y_inner_dim = inner_dim * repeats; \
        auto nthreads = outer_dim * repeat_dim * inner_dim * repeats; \
        _Repeat<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, repeat_dim, inner_dim, \
                y_inner_dim, x, y); \
    }

#define DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER(T) \
    template<> void RepeatGrad<T, CUDAContext>( \
        const int               outer_dim, \
        const int               repeat_dim, \
        const int               inner_dim, \
        const int               repeats, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        auto y_inner_dim = inner_dim * repeats; \
        auto nthreads = outer_dim * repeat_dim * inner_dim; \
        _RepeatGrad<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, repeat_dim, inner_dim, \
                y_inner_dim, repeats, dy, dx); \
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
    const int               repeat_dim,
    const int               inner_dim,
    const int               repeats,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    auto y_inner_dim = inner_dim * repeats;
    auto nthreads = outer_dim * repeat_dim * inner_dim;
    _RepeatGradHalf
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
             0, ctx->cuda_stream() >> > \
        (nthreads, repeat_dim, inner_dim,
            y_inner_dim, repeats,
                reinterpret_cast<const half*>(dy),
                    reinterpret_cast<half*>(dx));
}

#undef DEFINE_REPEAT_KERNEL_LAUNCHER
#undef DEFINE_REPEAT_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA