#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Arange <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Arange(
    const int               count,
    const int               start,
    const int               step,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = start + idx * step;
    }
}

/*! Arange <T = float32, Device = CUDA> */

template<> void Arange<float, CUDAContext>(
    const int               count,
    const int               start,
    const int               step,
    float*                  y,
    CUDAContext*            ctx) {
    _Arange<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, start, step, y);
}

/*! Arange <T = int32, Device = CUDA> */

template<> void Arange<int, CUDAContext>(
    const int               count,
    const int               start,
    const int               step,
    int*                    y,
    CUDAContext*            ctx) {
    _Arange<int>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, start, step, y);
}

/*! Arange <T = int64, Device = CUDA> */

template<> void Arange<int64_t, CUDAContext>(
    const int               count,
    const int               start,
    const int               step,
    int64_t*                y,
    CUDAContext*            ctx) {
    _Arange<int64_t>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, start, step, y);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA