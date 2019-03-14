#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Slice <T = ?, Device = CPU> */

template <typename T>
void _Slice(
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    int64_t x_offset; int64_t cols = y_slice_dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        ctx->Copy<T, CPUContext, CPUContext>(
            cols, y + n * cols, x + x_offset);
    }
}

/*! SliceGrad <T = ?, Device = CPU> */

template <typename T>
void _SliceGrad(
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    int64_t x_offset; int64_t cols = y_slice_dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        if (dy != nullptr) {
            ctx->Copy<T, CPUContext, CPUContext>(
                cols, dx + x_offset, dy + n * cols);
        } else {
            ctx->Memset(sizeof(T) * cols, dx + x_offset);
        }
    }
}

/*! Kernel Launchers */

#define DEFINE_SLICE_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               x_slice_dim, \
        const int               y_slice_dim, \
        const int               slice_offset, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name(outer_dim, inner_dim, x_slice_dim, \
            y_slice_dim, slice_offset, x, y, ctx); \
    }

DEFINE_SLICE_KERNEL_LAUNCHER(Slice, bool);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, int8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, uint8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, int);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, int64_t);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, float16);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, float);
DEFINE_SLICE_KERNEL_LAUNCHER(Slice, double);

DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, bool);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, int8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, uint8_t);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, int);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, int64_t);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, float16);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, float);
DEFINE_SLICE_KERNEL_LAUNCHER(SliceGrad, double);

#undef DEFINE_SLICE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon