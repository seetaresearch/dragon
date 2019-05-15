#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/norm/l2_norm_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    num_axes_ = OpArg<int64_t>("num_axes", -1); \
    if (axis_ < 0) axis_ += X.ndim(); \
    if (num_axes_ < 0) num_axes_ = X.ndim() - axis_; \
    else if (num_axes_ == 0) num_axes_ = 1; \
    end_axis_ = axis_ + num_axes_; \
    CHECK(axis_ >= 0 && end_axis_ <= X.ndim())

template <class Context> template <typename T>
void L2NormOp<Context>::RunImpl() {
    auto scale = mode_ == "MEAN" ?
        1.f / reduce_dim_ : 1.f;
    auto nelements = outer_dim_ * inner_dim_;

    auto* norm = ws()
        ->CreateTensor(unique_name("rnorm"))
        ->Reshape({ nelements })
        ->template mutable_data<T, Context>();

    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    // Compute inversed ||norm||2
    //    = \rsqrt(\sum_{i} x_{i,j}^{2} + eps)
    math::Square(Y(0)->count(), x, y, ctx());

    vec32_t dims = {
        (int)outer_dim_,
        (int)reduce_dim_,
        (int)inner_dim_
    }, axes = { 1 };

    kernel::ReduceSum(
        3, dims.data(),
        1, axes.data(),
        scale, y,
        norm, ctx()
    );

    math::InvStd(nelements, eps_, norm, norm, ctx());

    // Affine
    kernel::Repeat(
        outer_dim_,
        inner_dim_,
        1,
        reduce_dim_,
        norm, y, ctx()
    );
    math::Mul(
        Y(0)->count(),
        x, y,
        y, ctx()
    );
}

template <class Context>
void L2NormOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    // Reduce along [axis, end_axis)
    outer_dim_ = X(0).count(0, axis_);
    reduce_dim_ = X(0).count(axis_, end_axis_);
    inner_dim_ = X(0).count(end_axis_);

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void L2NormGradientOp<Context>::RunImpl() {
    auto scale = mode_ == "MEAN" ?
        1.f / reduce_dim_ : 1.f;
    auto nelements = outer_dim_ * inner_dim_;

    auto buf = ws()
        ->template data<T, Context>
            ({ nelements, Y(0)->count() });

    auto* norm = ws()
        ->GetTensor(unique_name("rnorm"))
        ->template mutable_data<T, Context>();

    auto* x  = X(0).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    // Compute A:
    //    = X \dot \sum_{i} x_{i,j}dy_{i,j}
    math::Mul(Y(0)->count(), x, dy, dx, ctx());

    vec32_t dims = {
        (int)outer_dim_,
        (int)reduce_dim_,
        (int)inner_dim_
    }, axes = { 1 };

    kernel::ReduceSum(
        3, dims.data(),
        1, axes.data(),
        scale, dx,
        buf[0], ctx()
    );
    kernel::Repeat(
        outer_dim_,
        inner_dim_,
        1,
        reduce_dim_,
        buf[0], dx, ctx()
    );
    math::Mul(
        Y(0)->count(),
        x, dx,
        dx, ctx()
    );

    // Compute B:
    //    = A * (rnorm ** 2)
    math::Square(nelements, norm, buf[0], ctx());
    kernel::Repeat(
        outer_dim_,
        inner_dim_,
        1,
        reduce_dim_,
        buf[0], buf[1], ctx()
    );
    math::Mul(
        Y(0)->count(),
        dx, buf[1],
        dx, ctx()
    );

    // Compute Y:
    //    (dY - B) * rnorm
    math::Sub(Y(0)->count(), dy, dx, dx, ctx());
    kernel::Repeat(
        outer_dim_,
        inner_dim_,
        1,
        reduce_dim_,
        norm, buf[1], ctx()
    );
    math::Mul(
        Y(0)->count(),
        dx, buf[1],
        dx, ctx()
    );
}

template <class Context>
void L2NormGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    // Broadcast along [axis, end_axis)
    outer_dim_ = X(0).count(0, axis_);
    reduce_dim_ = X(0).count(axis_, end_axis_);
    inner_dim_ = X(0).count(end_axis_);

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(L2Norm);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2Norm);
#endif

DEPLOY_CPU(L2NormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2NormGradient);
#endif

OPERATOR_SCHEMA(L2Norm)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(L2NormGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(L2Norm, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon