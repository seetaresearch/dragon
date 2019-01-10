#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! AdamUpdate <T = float32, Device = CUDA> */

template <typename T>
__global__ void _AdamUpdate(
    const int               count,
    const T                 lr,
    const T                 beta1,
    const T                 beta2,
    const T                 eps,
    T*                      g,
    T*                      m,
    T*                      v) {
    CUDA_1D_KERNEL_LOOP(i, count) {
        T gi = g[i];
        T mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
        T vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        g[i] = lr * mi / (sqrt(vi) + eps);
    }
}

template <> void AdamUpdate<float, CUDAContext>(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    float*                  g,
    float*                  m,
    float*                  v,
    CUDAContext*            ctx) {
    _AdamUpdate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, lr, beta1, beta2, eps, g, m, v);
}

/*! AdamUpdate <T = float16, Device = CUDA> */

__global__ void _AdamUpdateHalf(
    const int               count,
    const half              lr,
    const half              beta1,
    const half              beta2,
    const half              eps,
    half*                   g,
    half*                   m,
    half*                   v) {
    CUDA_1D_KERNEL_LOOP(i, count) {
#if __CUDA_ARCH__ >= 530
        half gi = g[i];
        half kOne = __float2half(1.f);
        half mi = m[i] = __hadd(
            __hmul(m[i], beta1),
            __hmul(gi, __hsub(kOne, beta1))
        );
        half vi = v[i] = __hadd(
            __hmul(v[i], beta2),
            __hmul(gi, __hmul(gi, __hsub(kOne, beta2)))
        );
        g[i] = __hdiv(
            __hmul(lr, mi),
            __hadd(hsqrt(vi), eps)
        );
#endif
    }
}

template <> void AdamUpdate<float16, CUDAContext>(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    float16*                g,
    float16*                m,
    float16*                v,
    CUDAContext*            ctx) {
    _AdamUpdateHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, cast::to<half>(lr),
            cast::to<half>(beta1),
                cast::to<half>(beta2),
                    cast::to<half>(eps),
                        reinterpret_cast<half*>(g),
                            reinterpret_cast<half*>(m),
                                reinterpret_cast<half*>(v));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA