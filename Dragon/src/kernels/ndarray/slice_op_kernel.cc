#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Slice <T = float32, Device = CPU> */

template <> void Slice<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        y_offset = n * y_slice_dim * inner_dim;
        ctx->Copy<float, CPUContext, CPUContext>(
            y_slice_dim * inner_dim, y + y_offset, x + x_offset);
    }
}

/*! SliceGrad <T = float32, Device = CPU> */

template <> void SliceGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        y_offset = n * y_slice_dim * inner_dim;
        ctx->Copy<float, CPUContext, CPUContext>(
            y_slice_dim * inner_dim, dx + x_offset, dy + y_offset);
    }
}

}  // namespace kernel

}  // namepsace dragon