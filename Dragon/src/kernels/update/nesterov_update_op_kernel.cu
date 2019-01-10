#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! NesterovUpdate <T = float32, Device = CUDA> */

template <typename T>
__global__ void _NesterovUpdate(
    const int               count,
    const T                 lr,
    const T                 momentum,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T hi = h[i];
        T hi_new = h[i] = momentum * hi + lr * g[i];
        g[i] = (1 + momentum) * hi_new - momentum * hi;
    }
}

template <> void NesterovUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _NesterovUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, lr, momentum, g, h);
}

/*! NesterovUpdate <T = float16, Device = CUDA> */

__global__ void _NesterovUpdateHalf(
    const int               count,
    const half              lr,
    const half              momentum,
    half*                   g,
    half*                   h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half hi = h[i];
        half hi_new = h[i] = __hadd(
            __hmul(momentum, hi),
            __hmul(lr, g[i])
        );
        half kOne = __float2half(1.f);
        g[i] = __hsub(
            __hmul(__hadd(kOne, momentum), hi_new),
            __hmul(momentum, hi)
        );
#endif
    }
}

__global__ void _NesterovUpdateHalf2(
    const int               count,
    const half2             lr,
    const half2             momentum,
    half2*                  g,
    half2*                  h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half2 hi = h[i];
        half2 hi_new = h[i] = __hadd2(
            __hmul2(momentum, hi),
            __hmul2(lr, g[i])
        );
        half2 kOne = __float2half2_rn(1.f);
        g[i] = __hsub2(
            __hmul2(__hadd2(kOne, momentum), hi_new),
            __hmul2(momentum, hi)
        );
#endif
    }
}

template <> void NesterovUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float16*                g,
    float16*                h,
    CUDAContext*            ctx) {
    if ((count & 1) == 0) {
        _NesterovUpdateHalf2
            << < CUDA_BLOCKS(count >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count >> 1, cast::to<half2>(lr),
                cast::to<half2>(momentum),
                    reinterpret_cast<half2*>(g),
                        reinterpret_cast<half2*>(h));
    } else {
        _NesterovUpdateHalf
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, cast::to<half>(lr),
                cast::to<half>(momentum),
                    reinterpret_cast<half*>(g),
                        reinterpret_cast<half*>(h));
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA