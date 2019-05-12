#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template<> void Softmax<float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            multiplier,
    const float*            x,
    float*                  scale,
    float*                  y,
    CPUContext*             ctx) {
    const int dim = axis_dim * inner_dim;
    for (int i = 0; i < outer_dim; ++i) {
       ctx->Copy<float, CPUContext, CPUContext>(
            inner_dim, scale, x + i * dim);
        for (int j = 0; j < axis_dim; ++j) {
            for (int k = 0; k < inner_dim; k++)
                scale[k] = std::max(
                    scale[k], x[i * dim + j * inner_dim + k]
                );
        }
        math::Gemm(
            CblasNoTrans,
            CblasNoTrans,
            axis_dim, inner_dim, 1,
            -1.f, multiplier, scale,
            1.f, y, ctx
        );
        math::Exp(dim, y, y, ctx);
        math::Gemv(
            CblasTrans,
            axis_dim, inner_dim,
            1.f, y, multiplier,
            0.f, scale, ctx
        );
        for (int j = 0; j < axis_dim; ++j) {
            math::Div(inner_dim, y, scale, y, ctx);
            y += inner_dim;
        }
    }
}

template <typename T>
void _SoftmaxGrad(
    const int           outer_dim,
    const int           axis_dim,
    const int           inner_dim,
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

/* <T = float32, Device = CPU> */

template<> void SoftmaxGrad<float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float*            multiplier,
    const float*            dy,
    const float*            y,
    float*                  scale,
    float*                  dx,
    CPUContext*             ctx) {
    _SoftmaxGrad(
        outer_dim, axis_dim,
        inner_dim, dy, y, dx
    );
}

}  // namespace kernel

}  // namepsace dragon