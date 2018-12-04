#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! OneHot <T = ?, Device = CUDA> */

template <typename T>
__global__ void _OneHot(
    const int               count,
    const int               depth,
    const int               on_value,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int val = x[idx];
        y[idx * depth + val] = on_value;
    }
}

/*! OneHot <T = float32, Device = CUDA> */

template <> void OneHot<float, CUDAContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _OneHot<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, depth, on_value, x, y);
}

/*! OneHot <T = int32, Device = CUDA> */

template <> void OneHot<int, CUDAContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int*              x,
    int*                    y,
    CUDAContext*            ctx) {
    _OneHot<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, depth, on_value, x, y);
}

/*! OneHot <T = int64, Device = CUDA> */

template <> void OneHot<int64_t, CUDAContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int64_t*          x,
    int64_t*                y,
    CUDAContext*            ctx) {
    _OneHot<int64_t>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, depth, on_value, x, y);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA