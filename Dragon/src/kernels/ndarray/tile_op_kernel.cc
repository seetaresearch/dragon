#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Tile <T = ?, Device = CPU> */

template <typename T>
void _Tile(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y) {
    vector<int> index(ndims, 0); int x_idx;
    for (int y_idx = 0; y_idx < nthreads; ++y_idx) {
        x_idx = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            x_idx += (index[d] % x_dims[d]) * x_strides[d];
        }
        y[y_idx] = x[x_idx];
        utils::IncreaseIndexInDims(ndims, y_dims, index.data());
    }
}

/*! TileGrad <T = ?, Device = CPU> */

template <typename T>
void _TileGrad(
    const int               rows,
    const int               cols,
    const int               multiple,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    for (int i = 0; i < rows; ++i) {
        ctx->Copy<T, CPUContext, CPUContext>(
            cols, dx, dy); dy += cols;
        for (int m = 1; m < multiple; ++m) {
            math::Add<T, CPUContext>(
                cols, dy, dx, dx, ctx); dy += cols;
        } dx += cols;
    }
}

/*! Kernel Launchers */

#define DEFINE_TILE_KERNEL_LAUNCHER(T) \
    template<> void Tile<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _Tile<T>(count, ndims, x_dims, x_strides, y_dims, x, y); \
    }

#define DEFINE_TILE_GRAD_KERNEL_LAUNCHER(T) \
    template<> void TileGrad<T, CPUContext>( \
        const int               rows, \
        const int               cols, \
        const int               multiple, \
        const T*                dy, \
        T*                      dx, \
        CPUContext*             ctx) { \
        _TileGrad<T>(rows, cols, multiple, dy, dx, ctx); \
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
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_TILE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_TILE_KERNEL_LAUNCHER
#undef DEFINE_TILE_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon