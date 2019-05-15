#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/exp_op.h"

namespace dragon {

template <class Context> template <typename T>
void ExpOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Exp(Y(0)->count(), x, y, ctx());
}

template <class Context>
void ExpOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void ExpGradientOp<Context>::RunImpl() {
    auto* y = X(0).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    math::Mul(Y(0)->count(), dy, y, dx, ctx());
}

template <class Context>
void ExpGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Exp);
#ifdef WITH_CUDA
DEPLOY_CUDA(Exp);
#endif

DEPLOY_CPU(ExpGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpGradient);
#endif

OPERATOR_SCHEMA(Exp)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ExpGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Exp, InplaceGradientMaker);

}  // namespace dragon