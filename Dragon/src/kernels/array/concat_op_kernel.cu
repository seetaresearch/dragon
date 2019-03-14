#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Concat <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Concat(
    const int               nthreads,
    const int               inner_dim,
    const int               x_cols,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        const int outer_idx = x_idx / x_cols;
        const int concat_idx = x_idx % x_cols;
        const int y_idx = (outer_idx * y_concat_dim + concat_offset)
                                * inner_dim + concat_idx;
        y[y_idx] = x[x_idx];
    }
}

/*! Kernel Launchers */

#define DEFINE_CONCAT_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               x_concat_dim, \
        const int               y_concat_dim, \
        const int               concat_offset, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto x_cols = x_concat_dim * inner_dim; \
        auto nthreads = outer_dim * x_concat_dim * inner_dim; \
        _##name<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, inner_dim, x_cols, \
                y_concat_dim, concat_offset, x, y); \
    }

DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, bool);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, int8_t);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, uint8_t);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, int);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, int64_t);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, float16);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, float);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, double);

#undef DEFINE_CONCAT_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA