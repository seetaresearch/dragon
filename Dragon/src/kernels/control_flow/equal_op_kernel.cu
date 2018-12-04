  #ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Equal <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Equal(
    const int               count,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = fabs(a[idx] - b[idx]) < FLT_EPSILON ? (T)1 : (T)0;
    }
}

template <> void Equal<float, CUDAContext>(
    const int               count,
    const float*            a,
    const float*            b,
    float*                  y,
    CUDAContext*            ctx) {
    _Equal<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, a, b, y);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA