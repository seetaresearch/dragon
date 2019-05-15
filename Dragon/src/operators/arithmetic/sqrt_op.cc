#include "utils/math_functions.h"
#include "operators/arithmetic/sqrt_op.h"

namespace dragon {

template <class Context> template <typename T>
void SqrtOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Sqrt(X(0).count(), x, y, ctx());
}

template <class Context>
void SqrtOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void SqrtGradientOp<Context>::RunImpl() {
    auto* y = X(0).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    math::Scale(X(0).count(), 2.f, y, dx, ctx());
    math::Inv(X(0).count(), dx, dx, ctx());
}

template <class Context>
void SqrtGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Sqrt);
#ifdef WITH_CUDA
DEPLOY_CUDA(Sqrt);
#endif

DEPLOY_CPU(SqrtGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SqrtGradient);
#endif

OPERATOR_SCHEMA(Sqrt)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(SqrtGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Sqrt, InplaceGradientMaker);

}  // namespace dragon