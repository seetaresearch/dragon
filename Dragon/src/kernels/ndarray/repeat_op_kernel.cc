#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Repeat <T = float32, Device = CPU> */

template <> void Repeat<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < repeats; ++k) {
                ctx->Copy<float, CPUContext, CPUContext>(
                    inner_dim, y, x);
                y += inner_dim;
            }
            x += inner_dim;
        }
    }
}

/*! RepeatGrad <T = float32, Device = CPU> */

template <> void RepeatGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            ctx->Copy<float, CPUContext, CPUContext>(
                inner_dim, dx, dy);
            dy += inner_dim;
            for (int k = 1; k < repeats; ++k) {
                math::Axpy<float, CPUContext>(
                    inner_dim, 1.f, dy, dx, ctx);
                dy += inner_dim;
            }
            dx += inner_dim;
        }
    }
} 

}  // namespace kernel

}  // namepsace dragon