#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Transpose <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Transpose(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
       int x_idx = 0, y_idx = idx;
       for (int j = 0; j < ndim; ++j) {
           int k = order[j];
           x_idx += (y_idx / new_steps[j]) * old_steps[k];
           y_idx %= new_steps[j];
       }
       y[idx] = x[x_idx];
   }
}

template <> void Transpose<float, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _Transpose<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, ndim, order, old_steps, new_steps, x, y);
}

/*! Transpose <T = float16, Device = CUDA> */

template <typename T>
__global__ void _TransposeHalf(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
       int x_idx = 0, y_idx = idx;
       for (int j = 0; j < ndim; ++j) {
           int k = order[j];
           x_idx += (y_idx / new_steps[j]) * old_steps[k];
           y_idx %= new_steps[j];
       }
       y[idx] = x[x_idx];
   }
}

template <> void Transpose<float16, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _TransposeHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, ndim, order, old_steps, new_steps,
            reinterpret_cast<const half*>(x),
                reinterpret_cast<half*>(y));
}

/*! TransposeGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _TransposeGrad(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int x_idx = 0, y_idx = idx;
        for (int j = 0; j < ndim; ++j) {
            int k = order[j];
            x_idx += (y_idx / new_steps[j]) * old_steps[k];
            y_idx %= new_steps[j];
        }
        dx[x_idx] = dy[idx];
    }
}

template <> void TransposeGrad<float, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _TransposeGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, ndim, order, old_steps, new_steps, dy, dx);
}

/*! TransposeGrad <T = float16, Device = CUDA> */

template <typename T>
__global__ void _TransposeGradHalf(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int x_idx = 0, y_idx = idx;
        for (int j = 0; j < ndim; ++j) {
            int k = order[j];
            x_idx += (y_idx / new_steps[j]) * old_steps[k];
            y_idx %= new_steps[j];
        }
        dx[x_idx] = dy[idx];
    }
}

template <> void TransposeGrad<float16, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    _TransposeGradHalf<half>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, ndim, order, old_steps, new_steps,
            reinterpret_cast<const half*>(dy),
                reinterpret_cast<half*>(dx));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA