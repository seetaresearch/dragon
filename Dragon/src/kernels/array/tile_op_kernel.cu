#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

/*! Tile <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Tile(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        int x_idx = 0, tmp = y_idx;
#pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
            x_idx += r % __ldg(x_dims + d) * __ldg(x_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
            x_idx += r % x_dims[d] * x_strides[d];
#endif
        }
        y[y_idx] = x[x_idx];
    }
}

/*! TileGrad <T = ?, Device = CUDA> */

template <typename T>
__global__ void _TileGrad(
    const int               nthreads,
    const int               cols,
    const int               tiled_cols,
    const int               multiple,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        T gradient = 0;
        const int col_idx = x_idx % cols;
        const int row_idx = x_idx / cols;
        const int y_offset = row_idx * tiled_cols + col_idx;
        for (int m = 0; m < multiple; ++m)
            gradient += dy[y_offset + m * cols];
        dx[x_idx] = gradient;
    }
}

/*! TileGrad <T = float16, Device = CUDA> */

__global__ void _TileGradHalf(
    const int               nthreads,
    const int               cols,
    const int               tiled_cols,
    const int               multiple,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        float gradient = 0.f;
        const int col_idx = x_idx % cols;
        const int row_idx = x_idx / cols;
        const int y_offset = row_idx * tiled_cols + col_idx;
        for (int m = 0; m < multiple; ++m)
            gradient += __half2float(dy[y_offset + m * cols]);
        dx[x_idx] = __float2half(gradient);
#endif
    }
}

/*! Kernel Launchers */

#define DEFINE_TILE_KERNEL_LAUNCHER(T) \
    template<> void Tile<T, CUDAContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Tile<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, ndims, x_dims, x_strides, y_dims, x, y); \
    }

#define DEFINE_TILE_GRAD_KERNEL_LAUNCHER(T) \
    template<> void TileGrad<T, CUDAContext>( \
        const int               rows, \
        const int               cols, \
        const int               multiple, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        auto nthreads = rows * cols; \
        auto tiled_cols = multiple * cols; \
        _TileGrad<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, cols, tiled_cols, multiple, dy, dx); \
    }

DEFINE_TILE_KERNEL_LAUNCHER(bool);
DEFINE_TILE_KERNEL_LAUNCHER(int8_t);
DEFINE_TILE_KERNEL_LAUNCHER(uint8_t);
DEFINE_TILE_KERNEL_LAUNCHER(int);
DEFINE_TILE_KERNEL_LAUNCHER(int64_t);
DEFINE_TILE_KERNEL_LAUNCHER(float16);
DEFINE_TILE_KERNEL_LAUNCHER(float);
DEFINE_TILE_KERNEL_LAUNCHER(double);

DEFINE_TILE_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(int);
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(double);

template<> void TileGrad<float16, CUDAContext>(
    const int               rows,
    const int               cols,
    const int               multiple,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    auto nthreads = rows * cols;
    auto tiled_cols = multiple * cols;
    _TileGradHalf
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, cols, tiled_cols, multiple,
            reinterpret_cast<const half*>(dy),
                reinterpret_cast<half*>(dx));
}

#undef DEFINE_TILE_KERNEL_LAUNCHER
#undef DEFINE_TILE_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA