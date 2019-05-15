#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Slice(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const int               cols,
    const int               slice_ofs,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int outer_idx = yi / cols;
        const int slice_idx = yi % cols;
        const int xi = (
            outer_idx * axis_dim + slice_ofs
                ) * inner_dim + slice_idx;
        y[yi] = x[xi];
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _SliceGrad(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const int               cols,
    const int               slice_ofs,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int outer_idx = yi / cols;
        const int slice_idx = yi % cols;
        const int xi = (
            outer_idx * axis_dim + slice_ofs
                ) * inner_dim + slice_idx;
        dx[xi] = dy ? dy[yi] : T(0);
    }
}

/* Kernel Launchers */

#define DEFINE_SLICE_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               slice_dim, \
        const int               slice_ofs, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto cols = slice_dim * inner_dim; \
        auto nthreads = outer_dim * cols; \
        _##name \
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            nthreads, \
            inner_dim, \
            axis_dim, \
            cols, \
            slice_ofs, \
            x, y \
        ); \
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

/* <T = float16, Device = CUDA> */

template<> __global__ void _SliceGrad<half>(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const int               cols,
    const int               slice_ofs,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int outer_idx = yi / cols;
        const int slice_idx = yi % cols;
        const int xi = (
            outer_idx * axis_dim + slice_ofs
                ) * inner_dim + slice_idx;
        dx[xi] = dy ? dy[yi] : __float2half(0.f);
#endif
    }
}

template <> void SliceGrad<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               slice_dim,
    const int               slice_ofs,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    auto cols = slice_dim * inner_dim;
    auto nthreads = outer_dim * cols;
    _SliceGrad
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads,
        inner_dim,
        axis_dim,
        cols,
        slice_ofs,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y)
    );
}

#undef DEFINE_SLICE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA