#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! ConstPad1d <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ConstPad1d(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T                 value,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        const int d = ex_d - pad_l;
        y[idx] = (d < 0 || d >= dim) ? value :
            x[(o * dim + d) * inner_dim + i];
    }
}

template <> void ConstPad1d<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float             value,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _ConstPad1d<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, pad_l, value, x, y);
}

/*! ReflectPad1d <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ReflectPad1d(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        int d = ex_d - pad_l;
        d = max(d, -d);
        d = min(d, 2 * dim - d - 2);
        y[idx] = x[(o * dim + d) * inner_dim + i];
    }
}

template <> void ReflectPad1d<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _ReflectPad1d<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, pad_l, x, y);
}

/*! EdgePad1d <T = float32, Device = CUDA> */

template <typename T>
__global__ void _EdgePad1d(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        const int d = min(dim - 1, max(ex_d - pad_l, 0));
        y[idx] = x[(o * dim + d) * inner_dim + i];
    }
}

template <> void EdgePad1d<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    _EdgePad1d<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, pad_l, x, y);
}

/*! ConstPad1dGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ConstPad1dGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % dim + pad_l;
        const int o = idx / inner_dim / dim;
        dx[idx] = dy[(o * ex_dim + ex_d) * inner_dim + i];
    }
}

template <> void ConstPad1dGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ConstPad1dGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, pad_l, dy, dx);
}

/*! ReflectPad1dGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ReflectPad1dGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        int d = ex_d - pad_l;
        d = max(d, -d);
        d = min(d, 2 * dim - d - 2);
        atomicAdd(&dx[(o * dim + d) * inner_dim + i], dy[idx]);
    }
}

template <> void ReflectPad1dGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _ReflectPad1dGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, pad_l, dy, dx);
}

/*! EdgePad1dGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _EdgePad1dGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        const int d = min(dim - 1, max(ex_d - pad_l, 0));
        atomicAdd(&dx[(o * dim + d) * inner_dim + i], dy[idx]);
    }
}

template <> void EdgePad1dGrad<float, CUDAContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _EdgePad1dGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dim, ex_dim, inner_dim, pad_l, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA