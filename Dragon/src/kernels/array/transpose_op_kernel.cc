#include "utils/op_kernel.h"
#include "utils/math_utils.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Transpose(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y) {
    vec32_t index(ndims, 0); int xi;
    for (int yi = 0; yi < nthreads; ++yi) {
        xi = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            xi += index[d] * x_strides[d];
        }
        y[yi] = x[xi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _TransposeGrad(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                dy,
    T*                      dx) {
    vec32_t index(ndims, 0); int xi;
    for (int yi = 0; yi < nthreads; ++yi) {
        xi = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            xi += index[d] * x_strides[d];
        }
        dx[xi] = dy[yi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* Kernel Launchers */

#define DEFINE_TRANSPOSE_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name(count, ndims, x_strides, y_dims, x, y); \
    }

DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, bool);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, int8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, uint8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, int);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, int64_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, float16);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, float);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, double);

DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, bool);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, int8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, uint8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, int);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, int64_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, float16);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, float);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, double);

#undef DEFINE_TRANSPOSE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon