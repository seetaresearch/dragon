#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! RMSPropUpdate <T = float32, Device = CUDA> */

template <typename T>
__global__ void _RMSPropUpdate(
    const int               count,
    const T                 lr,
    const T                 decay,
    const T                 eps,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T gi = g[i];
        T hi = h[i] = decay * h[i] + (1 - decay) * gi * gi;
        g[i] = lr * g[i] / (sqrt(hi) + eps);
    }
}

template <> void RMSPropUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _RMSPropUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, lr, decay, eps, g, h);
}

/*! RMSPropUpdate <T = float16, Device = CUDA> */

__global__ void _RMSPropUpdateHalf(
    const int               count,
    const half              lr,
    const half              decay,
    const half              eps,
    half*                   g,
    half*                   h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half gi = g[i];
        half kOne = __float2half(1.f);
        half hi = h[i] = __hadd(
            __hmul(decay, h[i]),
            __hmul(__hmul(__hsub(kOne, decay), gi), gi)
        );
        g[i] = __hdiv(
            __hmul(lr, g[i]),
            __hadd(hsqrt(hi), eps)
        );
#endif
    }
}

template <> void RMSPropUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float16*                g,
    float16*                h,
    CUDAContext*            ctx) {
    _RMSPropUpdateHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, cast::to<half>(lr),
            cast::to<half>(decay),
                cast::to<half>(eps),
                    reinterpret_cast<half*>(g),
                        reinterpret_cast<half*>(h));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA