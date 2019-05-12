#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Tile(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y) {
    vec32_t index(ndims, 0); int xi;
    for (int yi = 0; yi < nthreads; ++yi) {
        xi = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            xi += (index[d] % x_dims[d]) * x_strides[d];
        }
        y[yi] = x[xi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* <T = ?, Device = CPU> */

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
            math::Add(cols, dy, dx, dx, ctx);
            dy += cols;
        } dx += cols;
    }
}

/* Kernel Launchers */

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
        _Tile( \
            count, \
            ndims, \
            x_dims, \
            x_strides, \
            y_dims, \
            x, y \
        ); \
    }

#define DEFINE_TILE_GRAD_KERNEL_LAUNCHER(T) \
    template<> void TileGrad<T, CPUContext>( \
        const int               rows, \
        const int               cols, \
        const int               multiple, \
        const T*                dy, \
        T*                      dx, \
        CPUContext*             ctx) { \
        _TileGrad( \
            rows, \
            cols, \
            multiple, \
            dy, dx, ctx \
        ); \
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