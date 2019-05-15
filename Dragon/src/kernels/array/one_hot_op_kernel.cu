#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _OneHot(
    const int               nthreads,
    const int               depth,
    const int               on_value,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int val = x[i];
        y[i * depth + val] = (T)on_value;
    }
}

/* <T = float32, Device = CUDA> */

template <> void OneHot<float, CUDAContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _OneHot
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, depth, on_value, x, y
    );
}

/* <T = int32, Device = CUDA> */

template <> void OneHot<int, CUDAContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int*              x,
    int*                    y,
    CUDAContext*            ctx) {
    _OneHot
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, depth, on_value, x, y
    );
}

/* <T = int64, Device = CUDA> */

template <> void OneHot<int64_t, CUDAContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int64_t*          x,
    int64_t*                y,
    CUDAContext*            ctx) {
    _OneHot
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, depth, on_value, x, y
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA