#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/*! Dropout <T = float32, Device = CUDA> */

template<typename T>
__global__ void _Dropout(
    const int               count,
    const uint32_t          thresh,
    const float             scale,
    const T*                x,
    const uint32_t*         mask32,
    uint8_t*                mask8,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        mask8[idx] = (mask32[idx] > thresh);
        y[idx] = x[idx] * mask8[idx] * scale;
    }
}

template<> void Dropout<float, CUDAContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float*            x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float*                  y,
    CUDAContext*            ctx) {
    math::RandomUniform<uint32_t, CUDAContext>(
        count, float(0), float(UINT_MAX), mask32, ctx);
    auto thresh = static_cast<uint32_t>(UINT_MAX * prob);
    _Dropout<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, thresh, scale, x, mask32, mask8, y);
}

/*! Dropout <T = float16, Device = CUDA> */

__global__ void _DropoutHalf(
    const int               count,
    const uint32_t          thresh,
    const half              scale,
    const half*             x,
    const uint32_t*         mask32,
    uint8_t*                mask8,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        mask8[idx] = (mask32[idx] > thresh);
        y[idx] = __hmul(__hmul(x[idx], scale),
            __float2half((float)mask8[idx]));
#endif
    }
}

template<> void Dropout<float16, CUDAContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float16*          x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float16*                y,
    CUDAContext*            ctx) {
    math::RandomUniform<uint32_t, CUDAContext>(
        count, float(0), float(UINT_MAX), mask32, ctx);
    auto thresh = static_cast<uint32_t>(UINT_MAX * prob);
    _DropoutHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, thresh, dragon_cast<half, float>(scale),
            reinterpret_cast<const half*>(x),
                mask32, mask8, reinterpret_cast<half*>(y));
}

/*! ApplyMask <Tx = float32, Tm = uint8, Device = CUDA> */

template <typename Tx, typename Tm>
__global__ void _ApplyMask(
    const int               count,
    const float             scale,
    const Tx*               x,
    const Tm*               mask,
    Tx*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = x[idx] * mask[idx] * scale;
    }
}

template <> void ApplyMask<float, uint8_t, CUDAContext>(
    const int               count,
    const float             scale,
    const float*            x,
    const uint8_t*          mask,
    float*                  y,
    CUDAContext*            ctx) {
    _ApplyMask<float, uint8_t>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, scale, x, mask, y);
}

/*! ApplyMask <Tx = float16, Tm = uint8, Device = CUDA> */

template <typename Tm>
__global__ void _ApplyMaskHalf(
    const int               count,
    const half              scale,
    const half*             x,
    const Tm*               mask,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hmul(__hmul(x[idx], scale),
            __float2half((float)mask[idx]));
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
    _ApplyMaskHalf<uint8_t>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dragon_cast<half, float>(scale),
            reinterpret_cast<const half*>(x),
                 mask, reinterpret_cast<half*>(y));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA