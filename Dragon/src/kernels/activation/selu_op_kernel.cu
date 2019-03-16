#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! SElu <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SElu(
    const int               count,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 350
        y[idx] = __ldg(x + idx) > 0 ?
            1.0507f * __ldg(x + idx) :
                1.7581f * (exp(__ldg(x + idx)) - 1);
#else
        y[idx] = x[idx] > 0 ?
            1.0507f * x[idx] :
                1.7581f * (exp(x[idx]) - 1);
#endif
    }
}

template<> void SElu<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _SElu<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, x, y);
}

/*! SElu <T = float16, Device = CUDA> */

template <> __global__ void _SElu<half>(
    const int               count,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const float x32 = __half2float(x[idx]);
        y[idx] = __float2half(x32 > 0 ?
            1.0507f * x32 : 1.7581f * (
                exp(x32) - 1));
#endif
    }
}

template<> void SElu<float16, CUDAContext>(
    const int               count,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _SElu<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, reinterpret_cast<const half*>(x),
            reinterpret_cast<half*>(y));
}

/*! SEluGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SEluGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 350
        dx[idx] = __ldg(y + idx) > 0 ?
            1.0507f * __ldg(dy + idx) :
                (1.7581f + __ldg(y + idx))
                    * __ldg(dy + idx);
#else
        dx[idx] = y[idx] > 0 ?
            1.0507f * dy[idx] :
                (1.7581f + y[idx])
                    * dy[idx];
#endif
    }
}

template<> void SEluGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _SEluGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dy, y, dx);
}

/*! SEluGrad <T = float16, Device = CUDA> */

template<> __global__ void _SEluGrad<half>(
    const int               count,
    const half*             dy,
    const half*             y,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        const float y32 = __half2float(y[idx]);
        dx[idx] = __float2half(
            y32 > 0 ? 1.0507f * __half2float(dy[idx]) :
                (1.7581f + y32) * __half2float(dy[idx]));
#endif
    }
}

template<> void SEluGrad<float16, CUDAContext>(
    const int               count,
    const float16*          dy,
    const float16*          y,
    float16*                dx,
    CUDAContext*            ctx) {
    _SEluGrad<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, reinterpret_cast<const half*>(dy),
            reinterpret_cast<const half*>(y),
                reinterpret_cast<half*>(dx));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA