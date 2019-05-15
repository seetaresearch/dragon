#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SElu(
    const int               nthreads,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
        y[i] = __ldg(x + i) > 0 ?
            1.0507f * __ldg(x + i) :
            1.7581f * (exp(__ldg(x + i)) - 1);
#else
        y[i] = x[i] > 0 ?
            1.0507f * x[i] :
            1.7581f * (exp(x[i]) - 1);
#endif
    }
}

template<> void SElu<float, CUDAContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _SElu
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, x, y
    );
}

/* <T = float16, Device = CUDA> */

template <> __global__ void _SElu<half>(
    const int               nthreads,
    const half*             x,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const float x32 = __half2float(x[i]);
        y[i] = __float2half(
            x32 > 0 ? 1.0507f * x32 :
                      1.7581f * (exp(x32) - 1)
        );
#endif
    }
}

template<> void SElu<float16, CUDAContext>(
    const int               count,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _SElu
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y)
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _SEluGrad(
    const int               nthreads,
    const T*                dy,
    const T*                y,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
        dx[i] = __ldg(y + i) > 0 ?
            1.0507f * __ldg(dy + i) :
           (1.7581f + __ldg(y + i)) * __ldg(dy + i);
#else
        dx[i] = y[i] > 0 ?
            1.0507f * dy[i] :
           (1.7581f + y[i]) * dy[i];
#endif
    }
}

template<> void SEluGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CUDAContext*            ctx) {
    _SEluGrad
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count, dy, y, dx
    );
}

/* <T = float16, Device = CUDA> */

template<> __global__ void _SEluGrad<half>(
    const int               nthreads,
    const half*             dy,
    const half*             y,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        const float y32 = __half2float(y[i]);
        dx[i] = __float2half(
            y32 > 0 ?
                1.0507f * __half2float(dy[i]) :
               (1.7581f + y32) * __half2float(dy[i])
        );
#endif
    }
}

template<> void SEluGrad<float16, CUDAContext>(
    const int               count,
    const float16*          dy,
    const float16*          y,
    float16*                dx,
    CUDAContext*            ctx) {
    _SEluGrad
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx)
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA