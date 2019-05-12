#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/softmax_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
       << "\nExcepted axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OpArg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void SoftmaxOp<Context>::RunImpl() {
    DECLARE_MULTIPLIER(multiplier, axis_dim_);
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    auto* scratch = ws()
        ->template data<T, Context>
            ({ X(0).count() })[0];

    math::Copy(X(0).count(), x, y, ctx());

    kernel::Softmax(
        outer_dim_,
        axis_dim_,
        inner_dim_,
        multiplier,
        x, scratch,
        y, ctx()
    );
}

template <class Context>
void SoftmaxOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    axis_dim_  = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

template <class Context> template <typename T>
void SoftmaxGradientOp<Context>::RunImpl() {
    DECLARE_MULTIPLIER(multiplier, axis_dim_);

    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    auto* scratch = ws()
        ->template data<T, Context>
            ({ X(0).count() })[0];

    math::Copy(X(0).count(), dy, dx, ctx());

    kernel::SoftmaxGrad(
        outer_dim_,
        axis_dim_,
        inner_dim_,
        multiplier,
        dy, y,
        scratch,
        dx, ctx()
    );
}

template <class Context>
void SoftmaxGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    axis_dim_  = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(Softmax);
#ifdef WITH_CUDA
DEPLOY_CUDA(Softmax);
#endif

DEPLOY_CPU(SoftmaxGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxGradient);
#endif

OPERATOR_SCHEMA(Softmax)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(SoftmaxGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Softmax, InplaceGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon