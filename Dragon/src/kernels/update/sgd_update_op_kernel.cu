#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SGDUpdate <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SGDUpdate(
    const int               count,
    const T                 lr,
    const T                 momentum,
    T*                      g,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T hi = h[i];
        g[i] = h[i] = momentum * hi + lr * g[i];
    }
}

template <> void SGDUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h,
    CUDAContext*            ctx) {
    _SGDUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, lr, momentum, g, h);
}

/*! SGDUpdate <T = float16, Device = CUDA> */

__global__ void _SGDUpdateHalf(
    const int               count,
    const half              lr,
    const half              momentum,
    half*                   g,
    half*                   h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half hi = h[i];
        g[i] = h[i] = __hadd(
            __hmul(momentum, hi),
            __hmul(lr, g[i])
        );
#endif
    }
}

__global__ void _SGDUpdateHalf2(
    const int               count,
    const half2             lr,
    const half2             momentum,
    half2*                  g,
    half2*                  h) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half2 hi = h[i];
        g[i] = h[i] = __hadd2(
            __hmul2(momentum, hi),
            __hmul2(lr, g[i])
        );
#endif
    }
}

template <> void SGDUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float16*                g,
    float16*                h,
    CUDAContext*            ctx) {
    if ((count & 1) == 0) {
        _SGDUpdateHalf2
            << < CUDA_BLOCKS(count >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count >> 1, dragon_cast<half2, float>(lr),
                dragon_cast<half2, float>(momentum),
                    reinterpret_cast<half2*>(g),
                        reinterpret_cast<half2*>(h));
    } else {
        _SGDUpdateHalf
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, dragon_cast<half, float>(lr),
                dragon_cast<half, float>(momentum),
                    reinterpret_cast<half*>(g),
                        reinterpret_cast<half*>(h));
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA