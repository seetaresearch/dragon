#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! MixedPrecisionUpdate <T = float16, Device = CUDA> */

__global__ void _MixedPrecisionUpdateHalf(
    const int               count,
    const float*            updates,
    half*                   w,
    half*                   g) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        const float master_wi =
            __half2float(w[i]) - updates[i];
        w[i] = __float2half(master_wi);
        g[i] = __float2half(0.f);
    }
}

template <> void MixedPrecisionUpdate<float16, CUDAContext>(
    const int               count,
    const float*            updates,
    float16*                w,
    float16*                g,
    CUDAContext*            ctx) {
    _MixedPrecisionUpdateHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, updates, reinterpret_cast<half*>(w),
            reinterpret_cast<half*>(g));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA