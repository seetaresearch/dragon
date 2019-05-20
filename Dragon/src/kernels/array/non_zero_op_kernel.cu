#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
    do {                                  \
        const auto n_copy = n;            \
        *q = n_copy / d;                  \
        *r = n_copy % d;                  \
    } while (0)

/* <T = ?, Device = CUDA> */

__global__ void _UnravelIndex(
    const int               nthreads,
    const int               ndims,
    const int*              dims,
    const int64_t*          x,
    int64_t*                y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        int tmp = x[i], d;
        int64_t* Y = y + i * ndims;
#pragma unroll
        for (d = ndims - 1; d >= 0; --d) {
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(dims + d), tmp, &tmp, (Y + d));
#else
            FIXED_DIVISOR_DIV_MOD(dims[d], tmp, &tmp, (Y + d));
#endif
        }
    }
}

template <> void UnravelIndex<CUDAContext>(
    const int               count,
    const int               ndims,
    const int*              dims,
    const int64_t*          x,
    int64_t*                y,
    CUDAContext*            ctx) {
    _UnravelIndex
        <<< CUDA_BLOCKS(count), CUDA_THREADS, \
            0, ctx->cuda_stream() >>>(
        count, ndims, dims, x, y
    );
}

#undef FIXED_DIVISOR_DIV_MOD

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA