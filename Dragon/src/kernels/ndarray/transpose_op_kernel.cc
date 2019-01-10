#include "utils/op_kernel.h"
#include "utils/math_utils.h"

namespace dragon {

namespace kernel {

/*! Transpose <T = ?, Device = CPU> */

template <typename T>
void _Transpose(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y) {
    vector<int> index(ndims, 0); int x_idx;
    for (int y_idx = 0; y_idx < nthreads; ++y_idx) {
        x_idx = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            x_idx += index[d] * x_strides[d];
        }
        y[y_idx] = x[x_idx];
        utils::IncreaseIndexInDims(ndims, y_dims, index.data());
    }
}

/*! TransposeGrad <T = ?, Device = CPU> */

template <typename T>
void _TransposeGrad(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                dy,
    T*                      dx) {
    vector<int> index(ndims, 0); int x_idx;
    for (int y_idx = 0; y_idx < nthreads; ++y_idx) {
        x_idx = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            x_idx += index[d] * x_strides[d];
        }
        dx[x_idx] = dy[y_idx];
        utils::IncreaseIndexInDims(ndims, y_dims, index.data());
    }
}

/*! Kernel Launchers */

#define DEFINE_TRANSPOSE_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name<T>(count, ndims, x_strides, y_dims, x, y); \
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