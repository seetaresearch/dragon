#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Concat(
    const int               nthreads,
    const int               inner_dim,
    const int               cols,
    const int               cat_dim,
    const int               cat_ofs,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        const int outer_idx = xi / cols;
        const int cat_idx = xi % cols;
        const int yi = (
            outer_idx * cat_dim + cat_ofs
                ) * inner_dim + cat_idx;
        y[yi] = x[xi];
    }
}

/* Kernel Launchers */

#define DEFINE_CONCAT_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               cat_dim, \
        const int               cat_ofs, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        auto cols = axis_dim * inner_dim; \
        auto nthreads = outer_dim * axis_dim * inner_dim; \
        _##name \
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            nthreads, \
            inner_dim, \
            cols, \
            cat_dim, \
            cat_ofs, \
            x, y \
        ); \
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