#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Concat <T = float32, Device = CPU> */

template <> void Concat<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        ctx->Copy<float, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, y + y_offset, x + x_offset);
    }
}

/*! Concat <T = float16, Device = CPU> */

template <> void Concat<float16, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        ctx->Copy<float16, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, y + y_offset, x + x_offset);
    }
}

/*! ConcatGrad <T = float32, Device = CPU> */

template <> void ConcatGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        ctx->Copy<float, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, dx + x_offset, dy + y_offset);
    }
}

/*! ConcatGrad <T = float16, Device = CPU> */

template <> void ConcatGrad<float16, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          dy,
    float16*                dx,
    CPUContext*             ctx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        ctx->Copy<float16, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, dx + x_offset, dy + y_offset);
    }
}

}  // namespace kernel

}  // namepsace dragon