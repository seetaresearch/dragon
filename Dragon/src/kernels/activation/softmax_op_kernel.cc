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


template <typename T>
void _SoftmaxGrad(
    const int           outer_dim,
    const int           inner_dim,
    const int           axis_dim,
    const T*            dy,
    const T*            y,
    T*                  dx) {
    int row_stride, axis_offset, idx;
    for (int i = 0; i < outer_dim; ++i) {
        row_stride = i * axis_dim * inner_dim;
        for (int j = 0; j < inner_dim; ++j) {
            T dYxY = 0; axis_offset = row_stride + j;
            for (int k = 0; k < axis_dim; ++k) {
                idx = axis_offset + k * inner_dim;
                dYxY += (dy[idx] * y[idx]);
            }
            for (int k = 0; k < axis_dim; ++k) {
                idx = axis_offset + k * inner_dim;
                dx[idx] = (dx[idx] - dYxY) * y[idx];
            }
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
    _SoftmaxGrad<float>(outer_dim,
        inner_dim, classes, dy, y, dx);
}

}  // namespace kernel

}  // namepsace dragon