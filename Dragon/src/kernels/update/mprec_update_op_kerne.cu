#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float16, Device = CUDA> */

__global__ void _MixedPrecL2DecayHalf(
    const int               nthreads,
    const float             alpha,
    const half*             w,
    float*                  dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        dx[i] += (__half2float(w[i]) * alpha);
#endif
    }
}

template <> void MixedPrecL2Decay<float16, CUDAContext>(
    const int               count,
    const float             alpha,
    const float16*          w,
    float*                  dx,
    CUDAContext*            ctx) {
    _MixedPrecL2DecayHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        alpha,
        reinterpret_cast<const half*>(w),
        dx
    );
}

/* <T = float16, Device = CUDA> */

__global__ void _MixedPrecUpdateHalf(
    const int               nthreads,
    const float*            updates,
    half*                   w) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        w[i] = __float2half(__half2float(
            w[i]) - updates[i]);
#endif
    }
}

template <> void MixedPrecUpdate<float16, CUDAContext>(
    const int               count,
    const float*            updates,
    float16*                w,
    CUDAContext*            ctx) {
    _MixedPrecUpdateHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count,
        updates,
        reinterpret_cast<half*>(w)
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA