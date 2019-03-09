#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! MixedPrecisionL2Decay <T = float16, Device = CUDA> */

__global__ void _MixedPrecisionL2DecayHalf(
    const int               count,
    const float             alpha,
    const half*             w,
    float*                  dx) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        dx[i] += (__half2float(w[i]) * alpha);
#endif
    }
}

template <> void MixedPrecisionL2Decay<float16, CUDAContext>(
    const int               count,
    const float             alpha,
    const float16*          w,
    float*                  dx,
    CUDAContext*            ctx) {
    _MixedPrecisionL2DecayHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, alpha, reinterpret_cast<const half*>(w), dx);
}

/*! MixedPrecisionUpdate <T = float16, Device = CUDA> */

__global__ void _MixedPrecisionUpdateHalf(
    const int               count,
    const float*            updates,
    half*                   w) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        w[i] = __float2half(__half2float(
            w[i]) - updates[i]);
#endif
    }
}

template <> void MixedPrecisionUpdate<float16, CUDAContext>(
    const int               count,
    const float*            updates,
    float16*                w,
    CUDAContext*            ctx) {
    _MixedPrecisionUpdateHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, updates, reinterpret_cast<half*>(w));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA