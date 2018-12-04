#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Tile <T = float32, Device = CPU> */

template <> void Tile<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int t = 0; t < multiple; ++t) {
            ctx->Copy<float, CPUContext, CPUContext>(
                ex_inner_dim, y, x);
            y += ex_inner_dim;
        }
        x += ex_inner_dim;
    }
}

/*! TileGrad <T = float32, Device = CPU> */

template <> void TileGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    for (int i = 0; i < outer_dim; ++i) {
        ctx->Copy<float, CPUContext, CPUContext>(
            ex_inner_dim, dx, dy);
        dy += ex_inner_dim;
        for (int t = 1; t < multiple; ++t) {
            math::Axpy<float, CPUContext>(
                ex_inner_dim, 1.f, dy, dx, ctx);
            dy += ex_inner_dim;
        }
        dx += ex_inner_dim;
    }
}

}  // namespace kernel

}  // namepsace dragon