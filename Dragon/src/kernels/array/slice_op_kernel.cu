#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Slice <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Slice(
    const int               nthreads,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_cols,
    const int               slice_offset,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int outer_idx = y_idx / y_cols;
        const int slice_idx = y_idx % y_cols;
        const int x_idx = (outer_idx * x_slice_dim + slice_offset)
                                * inner_dim + slice_idx;
        y[y_idx] = x[x_idx];
    }
}

/*! SliceGrad <T = ?, Device = CUDA> */

template <typename T>
__global__ void _SliceGrad(
    const int               nthreads,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_cols,
    const int               slice_offset,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int outer_idx = y_idx / y_cols;
        const int slice_idx = y_idx % y_cols;
        const int x_idx = (outer_idx * x_slice_dim + slice_offset)
                                * inner_dim + slice_idx;
        dx[x_idx] = dy ? dy[y_idx] : 0;
    }
}

/*! Kernel Launchers */

#define DEFINE_SLICE_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               x_slice_dim, \
        const int               y_slice_dim, \
        const int               slice_offset, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto y_cols = y_slice_dim * inner_dim; \
        auto nthreads = outer_dim * y_slice_dim * inner_dim; \
        _##name<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, inner_dim, x_slice_dim, \
                y_cols, slice_offset, x, y); \
    }

DEFINE_SLICE_KERNEL_LAUNCHER(Slice, bool);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, int8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, uint8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, int);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, int64_t);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, float16);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, float);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, double);

DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, bool);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, int8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, uint8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, int);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, int64_t);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, float);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, double);

/*! SliceGrad <T = float16, Device = CUDA> */

__global__ void _SliceGradHalf(
    const int               nthreads,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_cols,
    const int               slice_offset,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int outer_idx = y_idx / y_cols;
        const int slice_idx = y_idx % y_cols;
        const int x_idx = (outer_idx * x_slice_dim + slice_offset)
                                * inner_dim + slice_idx;
        dx[x_idx] = dy ? dy[y_idx] : __float2half(0.f);
#endif
    }
}

template <> void SliceGrad<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    auto y_cols = y_slice_dim * inner_dim;
    auto nthreads = outer_dim * y_slice_dim * inner_dim;
    _SliceGradHalf
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, inner_dim, x_slice_dim,
            y_cols, slice_offset,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
}

#undef DEFINE_SLICE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA