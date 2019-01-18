#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"
#include "utils/cub_device.h"

namespace dragon {

namespace kernel {

/*! Gather <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Gather(
    const int               nthreads,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int64_t*          indices,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int outer_idx = y_idx / inner_dim / y_slice_dim;
        const int inner_idx = y_idx % inner_dim;
#if __CUDA_ARCH__ >= 350
        int select_idx = __ldg(indices +
            ((y_idx / inner_dim) % y_slice_dim));
#else
        int select_idx = indices[
            (y_idx / inner_dim) % y_slice_dim];
#endif
        select_idx = select_idx >= 0 ?
            select_idx : select_idx + x_slice_dim;
        const int x_idx = (outer_idx * x_slice_dim + select_idx)
                                * inner_dim + inner_idx;
        y[y_idx] = x[x_idx];
    }
}

/*! GatherGrad <T = ?, Device = CUDA> */

template <typename T>
__global__ void _GatherGrad(
    const int               nthreads,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int outer_idx = i / inner_dim;
        const int inner_idx = i % inner_dim;
        for (int j = 0; j < y_slice_dim; ++j) {
#if __CUDA_ARCH__ >= 350
            int select_idx = __ldg(indices + j);
#else
            int select_idx = indices[j];
#endif
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + x_slice_dim;
            const int x_idx = (outer_idx * x_slice_dim + select_idx)
                                     * inner_dim + inner_idx;
            const int y_idx = (outer_idx * y_slice_dim + j)
                                 * inner_dim + inner_idx;
            dx[x_idx] += dy[y_idx];
        }
    }
}

/*! GatherGrad <T = float16, Device = CUDA> */

template <> __global__ void _GatherGrad<half>(
    const int               nthreads,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int64_t*          indices,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int outer_idx = i / inner_dim;
        const int inner_idx = i % inner_dim;
        for (int j = 0; j < y_slice_dim; ++j) {
            int select_idx = __ldg(indices + j);
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + x_slice_dim;
            const int x_idx = (outer_idx * x_slice_dim + select_idx)
                * inner_dim + inner_idx;
            const int y_idx = (outer_idx * y_slice_dim + j)
                * inner_dim + inner_idx;
            dx[x_idx] = __hadd(dx[x_idx], dy[y_idx]);
        }
#endif
    }
}

/*! Kernel Launchers */

#define DEFINE_GATHER_KERNEL_LAUNCHER(T) \
    template <> void Gather<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               x_slice_dim, \
        const int               y_slice_dim, \
        const int64_t*          indices, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto nthreads = outer_dim * y_slice_dim * inner_dim; \
        _Gather<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, inner_dim, x_slice_dim, \
                y_slice_dim, indices, x, y); \
    }

#define DEFINE_GATHER_GRAD_KERNEL_LAUNCHER(T) \
    template <> void GatherGrad<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               x_slice_dim, \
        const int               y_slice_dim, \
        const int64_t*          indices, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        auto nthreads = outer_dim * inner_dim; \
        _GatherGrad<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, inner_dim, x_slice_dim, \
                y_slice_dim, indices, dy, dx); \
    }

DEFINE_GATHER_KERNEL_LAUNCHER(bool);
DEFINE_GATHER_KERNEL_LAUNCHER(int8_t);
DEFINE_GATHER_KERNEL_LAUNCHER(uint8_t);
DEFINE_GATHER_KERNEL_LAUNCHER(int);
DEFINE_GATHER_KERNEL_LAUNCHER(int64_t);
DEFINE_GATHER_KERNEL_LAUNCHER(float16);
DEFINE_GATHER_KERNEL_LAUNCHER(float);
DEFINE_GATHER_KERNEL_LAUNCHER(double);

DEFINE_GATHER_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_GATHER_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_GATHER_GRAD_KERNEL_LAUNCHER(int);
DEFINE_GATHER_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_GATHER_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GATHER_GRAD_KERNEL_LAUNCHER(double);

template <> void GatherGrad<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int64_t*          indices,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * inner_dim;
    _GatherGrad<half>
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, inner_dim, x_slice_dim,
            y_slice_dim, indices,
                reinterpret_cast<const half*>(dy),
                    reinterpret_cast<half*>(dx));
}

#undef DEFINE_GATHER_KERNEL_LAUNCHER
#undef DEFINE_GATHER_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA