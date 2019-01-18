#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Gather <T = ?, Device = CPU> */

template <typename T>
void _Gather(
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int64_t*          indices,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    int64_t x_offset, select_idx;
    for (int n = 0; n < outer_dim; ++n) {
        for (int i = 0; i < y_slice_dim; ++i) {
            select_idx = indices[i];
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + x_slice_dim;
            x_offset = (n * x_slice_dim + select_idx) * inner_dim;
            ctx->Copy<T, CPUContext, CPUContext>(
                inner_dim, y, x + x_offset);
            y += inner_dim;
        }
    }
}

/*! GatherGrad <T = ?, Device = CPU> */

template <typename T>
void _GatherGrad(
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    int64_t x_offset, select_idx;
    for (int n = 0; n < outer_dim; ++n) {
        for (int i = 0; i < y_slice_dim; ++i) {
            select_idx = indices[i];
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + x_slice_dim;
            x_offset = (n * x_slice_dim + select_idx) * inner_dim;
            math::Add<T, CPUContext>(inner_dim,
                dy, dx + x_offset, dx + x_offset, ctx);
            dy += inner_dim;
        }
    }
}

/*! Kernel Launchers */

#define DEFINE_GATHER_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               x_slice_dim, \
        const int               y_slice_dim, \
        const int64_t*          indices, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name<T> \
            (outer_dim, inner_dim, x_slice_dim, \
                y_slice_dim, indices, x, y, ctx); \
    }

DEFINE_GATHER_KERNEL_LAUNCHER(Gather, bool);
DEFINE_GATHER_KERNEL_LAUNCHER(Gather, int8_t);
DEFINE_GATHER_KERNEL_LAUNCHER(Gather, uint8_t);
DEFINE_GATHER_KERNEL_LAUNCHER(Gather, int);
DEFINE_GATHER_KERNEL_LAUNCHER(Gather, int64_t);
DEFINE_GATHER_KERNEL_LAUNCHER(Gather, float16);
DEFINE_GATHER_KERNEL_LAUNCHER(Gather, float);
DEFINE_GATHER_KERNEL_LAUNCHER(Gather, double);

DEFINE_GATHER_KERNEL_LAUNCHER(GatherGrad, int8_t);
DEFINE_GATHER_KERNEL_LAUNCHER(GatherGrad, uint8_t);
DEFINE_GATHER_KERNEL_LAUNCHER(GatherGrad, int);
DEFINE_GATHER_KERNEL_LAUNCHER(GatherGrad, int64_t);
DEFINE_GATHER_KERNEL_LAUNCHER(GatherGrad, float16);
DEFINE_GATHER_KERNEL_LAUNCHER(GatherGrad, float);
DEFINE_GATHER_KERNEL_LAUNCHER(GatherGrad, double);

#undef DEFINE_GATHER_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon