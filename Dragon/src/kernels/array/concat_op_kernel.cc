#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Concat(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               cat_dim,
    const int               cat_ofs,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    int64_t y_ofs, cols = axis_dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        y_ofs = (
            n * cat_dim + cat_ofs
                ) * inner_dim;
        math::Copy(
            cols,
            x + n * cols,
            y + y_ofs, ctx
        );
    }
}

/* Kernel Launchers */

#define DEFINE_CONCAT_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               cat_dim, \
        const int               cat_ofs, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name( \
            outer_dim, \
            inner_dim, \
            axis_dim, \
            cat_dim, \
            cat_ofs, \
            x, y, ctx \
        ); \
    }

DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, bool);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, int8_t);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, uint8_t);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, int);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, int64_t);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, float16);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, float);
DEFINE_CONCAT_KERNEL_LAUNCHER(Concat, double);

#undef DEFINE_CONCAT_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon