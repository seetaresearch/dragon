#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template<typename T>
__global__ void _DropPath(
    const int               nthreads,
    const int               cols,
    const float             thresh,
    const T                 scale,
    const T*                x,
    const float*            mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
        y[i] = x[i] * (T)(
            __ldg(mask + (i / cols)) > thresh
                ) * scale;
#else
        y[i] = x[i] * (T)(
            mask[i / cols] > thresh
                ) * scale;
#endif
    }
}

template<> void DropPath<float, CUDAContext>(
    const int               rows,
    const int               cols,
    const float             scale,
    const float*            x,
    const float*            mask,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = rows * cols;
    auto thresh = 1.f - (1.f / scale);
    _DropPath
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads, cols, thresh, scale, x, mask, y
    );
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _DropPath<half>(
    const int               nthreads,
    const int               cols,
    const float             thresh,
    const half              scale,
    const half*             x,
    const float*            mask,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const uint8_t m =
            __ldg(mask + (i / cols)) > thresh;
        y[i] = __hmul(
            __hmul(x[i], scale),
            __float2half((float)(
                __ldg(mask + (i / cols)) > thresh
            ))
        );
#endif
    }
}

template<> void DropPath<float16, CUDAContext>(
    const int               rows,
    const int               cols,
    const float             scale,
    const float16*          x,
    const float*            mask,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = rows * cols;
    auto thresh = 1.f - (1.f / scale);
    _DropPath
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads, cols,
        thresh,
        cast::to<half>(scale),
        reinterpret_cast<const half*>(x),
        mask,
        reinterpret_cast<half*>(y)
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA