#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _IndexSelect(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    int64_t x_offset, select_idx;
    for (int n = 0; n < outer_dim; ++n) {
        for (int i = 0; i < num_indices; ++i) {
            select_idx = indices[i];
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + axis_dim;
            x_offset = (
                n * axis_dim + select_idx
                    ) * inner_dim;
            math::Copy(
                inner_dim,
                x + x_offset,
                y, ctx
            ); y += inner_dim;
        }
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _IndexSelectGrad(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    int64_t x_offset, select_idx;
    auto nelements = outer_dim * axis_dim * inner_dim;
    math::Set(nelements, cast::to<T>(0.f), dx, ctx);
    for (int n = 0; n < outer_dim; ++n) {
        for (int i = 0; i < num_indices; ++i) {
            select_idx = indices[i];
            select_idx = select_idx >= 0 ?
                select_idx : select_idx + axis_dim;
            x_offset = (
                n * axis_dim + select_idx
                    ) * inner_dim;
            math::Add(
                inner_dim,
                dy, dx + x_offset,
                dx + x_offset, ctx
            ); dy += inner_dim;
        }
    }
}

/* Kernel Launchers */

#define DEFINE_INDEX_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               num_indices, \
        const int64_t*          indices, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name( \
            outer_dim, inner_dim, \
            axis_dim, num_indices, \
            indices, x, y, ctx \
        ); \
    }

DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, bool);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, int8_t);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, uint8_t);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, int);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, int64_t);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, float16);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, float);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelect, double);

DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelectGrad, int8_t);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelectGrad, uint8_t);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelectGrad, int);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelectGrad, int64_t);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelectGrad, float16);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelectGrad, float);
DEFINE_INDEX_KERNEL_LAUNCHER(IndexSelectGrad, double);

#undef DEFINE_INDEX_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon