#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! CanonicalAxis <T = int32, Device = CPU> */

template <> void CanonicalAxis<int, CPUContext>(
    const int               count,
    const int               dim,
    int*                    y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) if (y[i] < 0) y[i] += dim;
}

/*! Gather <T = ?, Device = CPU> */

template <typename T>
void _Gather(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            ctx->Copy<T, CPUContext, CPUContext>(
                inner_dim, y + y_offset, x + x_offset);
        }
    }
}

/*! Gather <T = float32, Device = CPU> */

template <> void Gather<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    _Gather<float>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, x, y, ctx);
}

/*! Gather <T = int32, Device = CPU> */

template <> void Gather<int, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              x,
    int*                    y,
    CPUContext*             ctx) {
    _Gather<int>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, x, y, ctx);
}

/*! GatherGrad <T = ?, Device = CPU> */

template <typename T>
void _GatherGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            math::Add<T, CPUContext>(inner_dim,
                dy + y_offset, dx + x_offset, dx + x_offset, ctx);
        }
    }
}

/*! GatherGrad <T = float32, Device = CPU> */

template <> void GatherGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    _GatherGrad<float>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, dy, dx, ctx);
}

/*! GatherGrad <T = int32, Device = CPU> */

template <> void GatherGrad<int, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              dy,
    int*                    dx,
    CPUContext*             ctx) {
    _GatherGrad<int>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, dy, dx, ctx);
}

}  // namespace kernel

}  // namepsace dragon