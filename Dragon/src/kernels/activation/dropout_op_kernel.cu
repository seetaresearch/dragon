#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template<typename T>
__global__ void _Dropout(
    const int               nthreads,
    const uint32_t          thresh,
    const T                 scale,
    const T*                x,
    const uint32_t*         mask32,
    uint8_t*                mask8,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const T m = mask8[i] =
            mask32[i] > thresh;
        y[i] = x[i] * m * scale;
    }
}

template<> void Dropout<float, CUDAContext>(
    const int               count,
    const float             prob,
    const float             scale,
    const float*            x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float*                  y,
    CUDAContext*            ctx) {
    auto thresh = (uint32_t)(UINT_MAX * prob);
    math::RandomUniform(count, 0.f, 1.f, mask32, ctx);
    _Dropout
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        thresh,
        scale,
        x, mask32,
        mask8, y
    );
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _Dropout<half>(
    const int               nthreads,
    const uint32_t          thresh,
    const half              scale,
    const half*             x,
    const uint32_t*         mask32,
    uint8_t*                mask8,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const float m = mask8[i] =
            mask32[i] > thresh;
        y[i] = __hmul(
            __hmul(x[i], scale),
            __float2half(m)
        );
#endif
    }
}

template<> void Dropout<float16, CUDAContext>(
    const int               count,
    const float             prob,
    const float             scale,
    const float16*          x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float16*                y,
    CUDAContext*            ctx) {
    auto thresh = (uint32_t)(UINT_MAX * prob);
    math::RandomUniform(count, 0.f, 1.f, mask32, ctx);
    _Dropout
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        thresh,
        cast::to<half>(scale),
        reinterpret_cast<const half*>(x),
        mask32, mask8,
        reinterpret_cast<half*>(y)
    );
}

/* <Tx = float32, Tm = uint8, Device = CUDA> */

template <typename Tx, typename Tm>
__global__ void _ApplyMask(
    const int               nthreads,
    const float             scale,
    const Tx*               x,
    const Tm*               mask,
    Tx*                     y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = x[i] * (Tx)mask[i] * scale;
    }
}

template <> void ApplyMask<float, uint8_t, CUDAContext>(
    const int               count,
    const float             scale,
    const float*            x,
    const uint8_t*          mask,
    float*                  y,
    CUDAContext*            ctx) {
    _ApplyMask
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, scale, x, mask, y
    );
}

/* <Tx = float16, Tm = uint8, Device = CUDA> */

template <typename Tm>
__global__ void _ApplyMaskHalf(
    const int               nthreads,
    const half              scale,
    const half*             x,
    const Tm*               mask,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul(
            __hmul(x[i], scale),
            __float2half((float)mask[i])
        );
#endif
    }
}

template <> void ApplyMask<float16, uint8_t, CUDAContext>(
    const int               count,
    const float             scale,
    const float16*          x,
    const uint8_t*          mask,
    float16*                y,
    CUDAContext*            ctx) {
    _ApplyMaskHalf
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        cast::to<half>(scale),
        reinterpret_cast<const half*>(x),
        mask,
        reinterpret_cast<half*>(y)
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA