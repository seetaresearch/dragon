#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"
#include "utils/cub_device.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _IndexSelect(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int outer_idx = yi / inner_dim / num_indices;
        const int inner_idx = yi % inner_dim;
#if __CUDA_ARCH__ >= 350
        int select_idx = __ldg(indices +
            ((yi / inner_dim) % num_indices));
#else
        int select_idx = indices[
            (yi / inner_dim) % num_indices];
#endif
        select_idx = select_idx >= 0 ?
            select_idx : select_idx + axis_dim;
        const int xi = (
            outer_idx * axis_dim + select_idx
                ) * inner_dim + inner_idx;
        y[yi] = x[xi];
    }
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _IndexSelectGrad(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int outer_idx = i / inner_dim;
        const int inner_idx = i % inner_dim;
        for (int j = 0; j < num_indices; ++j) {
#if __CUDA_ARCH__ >= 350
            int select_idx = __ldg(indices + j);
#else
            int select_idx = indices[j];
#endif
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + axis_dim;
            const int xi = (
                outer_idx * axis_dim + select_idx
                    ) * inner_dim + inner_idx;
            const int yi = (
                outer_idx * num_indices + j
                    ) * inner_dim + inner_idx;
            dx[xi] += dy[yi];
        }
    }
}

/* <T = float16, Device = CUDA> */

template <> __global__ void _IndexSelectGrad<half>(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int outer_idx = i / inner_dim;
        const int inner_idx = i % inner_dim;
        for (int j = 0; j < num_indices; ++j) {
            int select_idx = __ldg(indices + j);
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + axis_dim;
            const int xi = (
                outer_idx * axis_dim + select_idx
                    ) * inner_dim + inner_idx;
            const int yi = (
                outer_idx * num_indices + j
                    ) * inner_dim + inner_idx;
            dx[xi] = __hadd(dx[xi], dy[yi]);
        }
#endif
    }
}

/* Kernel Launchers */

#define DEFINE_INDEX_KERNEL_LAUNCHER(T) \
    template <> void IndexSelect<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               num_indices, \
        const int64_t*          indices, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto nthreads = outer_dim * num_indices * inner_dim; \
        _IndexSelect \
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            nthreads, inner_dim, \
            axis_dim, num_indices, \
            indices, x, y \
        ); \
    }

#define DEFINE_INDEX_GRAD_KERNEL_LAUNCHER(T) \
    template <> void IndexSelectGrad<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               num_indices, \
        const int64_t*          indices, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        auto nthreads = outer_dim * inner_dim; \
        _IndexSelectGrad \
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            nthreads, inner_dim, \
            axis_dim, num_indices, \
            indices, dy, dx \
        ); \
    }

DEFINE_INDEX_KERNEL_LAUNCHER(bool);
DEFINE_INDEX_KERNEL_LAUNCHER(int8_t);
DEFINE_INDEX_KERNEL_LAUNCHER(uint8_t);
DEFINE_INDEX_KERNEL_LAUNCHER(int);
DEFINE_INDEX_KERNEL_LAUNCHER(int64_t);
DEFINE_INDEX_KERNEL_LAUNCHER(float16);
DEFINE_INDEX_KERNEL_LAUNCHER(float);
DEFINE_INDEX_KERNEL_LAUNCHER(double);

DEFINE_INDEX_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_INDEX_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_INDEX_GRAD_KERNEL_LAUNCHER(int);
DEFINE_INDEX_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_INDEX_GRAD_KERNEL_LAUNCHER(float);
DEFINE_INDEX_GRAD_KERNEL_LAUNCHER(double);

template <> void IndexSelectGrad<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * inner_dim;
    _IndexSelectGrad
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads, inner_dim,
        axis_dim, num_indices,
        indices,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx)
    );
}

#undef DEFINE_INDEX_KERNEL_LAUNCHER
#undef DEFINE_INDEX_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA