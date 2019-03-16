#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Relu <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Relu(
    const int               count,
    const T                 slope,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(x + idx) > 0 ?
            __ldg(x + idx) :
                __ldg(x + idx) * slope;
#else
        y[idx] = x[idx] > 0 ?
            x[idx] : x[idx] * slope;
#endif
    }
}

template<> void Relu<float, CUDAContext>(
    const int               count,
    const float             slope,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Relu<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, slope, x, y);
}

/*! Relu <T = float16, Device = CUDA> */

template <> __global__ void _Relu<half>(
    const int               count,
    const half              slope,
    const half*             x,
    half*                   y) {
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hgt(__ldg(x + idx), kZero) ?
            __ldg(x + idx) : __hmul(
                __ldg(x + idx), slope);
#endif
    }
}

template <> __global__ void _Relu<half2>(
    const int               count,
    const half2             slope,
    const half2*            x,
    half2*                  y) {
    const half2 kZero = __float2half2_rn(0.f);
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __hbgt2(__ldg(x + idx), kZero) ?
            __ldg(x + idx) : __hmul2(
                __ldg(x + idx), slope);
#endif
    }
}

template<> void Relu<float16, CUDAContext>(
    const int               count,
    const float             slope,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    if ((count & 1) == 0) {
        _Relu<half2>
            << < CUDA_BLOCKS(count >> 1), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count >> 1, cast::to<half2>(slope),
                reinterpret_cast<const half2*>(x),
                    reinterpret_cast<half2*>(y));
    } else {
        _Relu<half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, cast::to<half>(slope),
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    }
}

/*! ReluGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ReluGrad(
    const int               count,
    const float             slope,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        dx[idx] = __ldg(dy + idx) * (
            (__ldg(y + idx) > 0) +
                slope * (__ldg(y + idx) <= 0)
        );
#else
        dx[idx] = dy[idx] * (
            (y[idx] > 0) +
                slope * (y[idx] <= 0)
        );
#endif
    }
}

template<> void ReluGrad<float, CUDAContext>(
    const int               count,
    const float             slope,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _ReluGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, slope, dy, y, dx);
}

/*! ReluGrad <T = float16, Device = CUDA> */

template <> __global__ void _ReluGrad<half>(
    const int               count,
    const float             slope,
    const half*             dy,
    const half*             y,
    half*                   dx) {
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        dx[idx] = __hmul(__ldg(dy + idx), __float2half(
            __hgt(__ldg(y + idx), kZero) +
                slope * __hle(__ldg(y + idx), kZero))
        );
#endif
    }
}

template<> void ReluGrad<float16, CUDAContext>(
    const int               count,
    const float             slope,
    const float16*          dy,
    const float16*          y,
    float16*                dx,
    CUDAContext*            ctx) {
    _ReluGrad<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, slope, reinterpret_cast<const half*>(dy),
            reinterpret_cast<const half*>(y),
                reinterpret_cast<half*>(dx));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA