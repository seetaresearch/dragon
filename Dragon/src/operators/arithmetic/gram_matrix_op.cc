#include "utils/math_functions.h"
#include "operators/arithmetic/gram_matrix_op.h"

namespace dragon {

template <class Context> template <typename T>
void GramMatrixOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    for (int i = 0; i < outer_dim_; i++) {
        math::Gemm(
            CblasNoTrans,
            CblasTrans,
            axis_dim_,
            axis_dim_,
            inner_dim_,
            1.f, x, x,
            0.f, y, ctx()
        );
        x += x_ofs_;
        y += y_ofs_;
    }
}

template <class Context>
void GramMatrixOp<Context>::RunOnDevice() {
    axis_dim_  = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    x_ofs_ = axis_dim_ * inner_dim_;
    y_ofs_ = axis_dim_ * axis_dim_;

    Y(0)->Reshape(
        { outer_dim_, axis_dim_, axis_dim_ }
    );

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void GramMatrixGradientOp<Context>::RunImpl() {
    auto* x  = X(0).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    for (int i = 0; i < outer_dim_; i++) {
        math::Gemm(
            CblasNoTrans,
            CblasNoTrans,
            axis_dim_,
            inner_dim_,
            axis_dim_,
            2.f, dy, x,
            0.f, dx, ctx()
        );
        dy += y_ofs_;
        dx += x_ofs_;
    }
}

template <class Context>
void GramMatrixGradientOp<Context>::RunOnDevice() {
    axis_dim_  = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    x_ofs_ = axis_dim_ * inner_dim_;
    y_ofs_ = axis_dim_ * axis_dim_;

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(GramMatrix);
#ifdef WITH_CUDA
DEPLOY_CUDA(GramMatrix);
#endif

DEPLOY_CPU(GramMatrixGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(GramMatrixGradient);
#endif

OPERATOR_SCHEMA(GramMatrix)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GramMatrixGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(GramMatrix, SimpleGradientMaker);

}  // namespace dragon