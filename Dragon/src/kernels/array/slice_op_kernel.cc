#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Slice(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               slice_dim,
    const int               slice_ofs,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    int64_t x_ofs, cols = slice_dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        x_ofs = (
            n * axis_dim + slice_ofs
                ) * inner_dim;
        math::Copy(
            cols,
            x + x_ofs,
            y + n * cols, ctx
        );
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _SliceGrad(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               slice_dim,
    const int               slice_ofs,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    int64_t x_ofs, cols = slice_dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        x_ofs = (
            n * axis_dim + slice_ofs
                ) * inner_dim;
        if (dy != nullptr) {
            math::Copy(
                cols,
                dy + n * cols,
                dx + x_ofs, ctx
            );
        } else {
            ctx->Memset(
                sizeof(T) * cols,
                dx + x_ofs
            );
        }
    }
}

/* Kernel Launchers */

#define DEFINE_SLICE_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               slice_dim, \
        const int               slice_ofs, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name( \
            outer_dim, \
            inner_dim, \
            axis_dim, \
            slice_dim, \
            slice_ofs, \
            x, y, ctx \
        ); \
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