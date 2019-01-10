#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Concat <T = ?, Device = CPU> */

template <typename T>
void _Concat(
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    int64_t y_offset = 0, x_cols = x_concat_dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        ctx->Copy<T, CPUContext, CPUContext>(
            x_cols, y + y_offset, x + n * x_cols);
    }
}

/*! Kernel Launchers */

#define DEFINE_CONCAT_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               x_concat_dim, \
        const int               y_concat_dim, \
        const int               concat_offset, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name(outer_dim, inner_dim, x_concat_dim, \
            y_concat_dim, concat_offset, x, y, ctx); \
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