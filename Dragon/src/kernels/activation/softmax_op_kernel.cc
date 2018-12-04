#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Softmax <T = float32, Device = CPU> */

template<> void Softmax<float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            sum_multiplier,
    const float*            x,
    float*                  scale,
    float*                  y,
    CPUContext*             ctx) {
    const int dim = count / outer_dim;
    for (int i = 0; i < outer_dim; ++i) {
       ctx->Copy<float, CPUContext, CPUContext>(
            inner_dim, scale, x + i * dim);
        for (int j = 0; j < classes; ++j) {
            for (int k = 0; k < inner_dim; k++)
                scale[k] = std::max(
                    scale[k], x[i * dim + j * inner_dim + k]
                );
        }
        math::Gemm<float, CPUContext>(
            CblasNoTrans, CblasNoTrans,
                classes, inner_dim, 1,
                    -1.f, sum_multiplier, scale, 1.f, y, ctx);
        math::Exp<float, CPUContext>(dim, y, y, ctx);
        math::Gemv<float, CPUContext>(
            CblasTrans, classes, inner_dim,
                1.f, y, sum_multiplier,
                    0.f, scale, ctx);
        for (int j = 0; j < classes; ++j) {
            math::Div<float, CPUContext>(inner_dim, y, scale, y, ctx);
            y += inner_dim;
        }
    }
}

/*! SoftmaxGrad <T = float32, Device = CPU> */

template<> void SoftmaxGrad<float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            sum_multiplier,
    const float*            dy,
    const float*            y,
    float*                  scale,
    float*                  dx,
    CPUContext*             ctx) {
    const int dim = count / outer_dim;
    for (int i = 0; i < outer_dim; ++i) {
        for (int k = 0; k < inner_dim; ++k)
            math::StridedDot<float, CPUContext>(classes,
                dx + i * dim + k, inner_dim,
                    y + i * dim + k, inner_dim, scale + k, ctx);
         math::Gemm<float, CPUContext>(
             CblasNoTrans, CblasNoTrans,
                classes, inner_dim, 1,
                    -1.f, sum_multiplier, scale,
                        1.f, dx + i * dim, ctx);
    }
    math::Mul<float, CPUContext>(count, dx, y, dx, ctx);
}

}  // namespace kernel

}  // namepsace dragon