#include "utils/math_functions.h"
#include "operators/arithmetic/log_op.h"

namespace dragon {

template <class Context> template <typename T>
void LogOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Log(X(0).count(), x, y, ctx());
}

template <class Context>
void LogOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void LogGradientOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    math::Div(X(0).count(), dy, x, dx, ctx());
}

template <class Context>
void LogGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Log);
#ifdef WITH_CUDA
DEPLOY_CUDA(Log);
#endif

DEPLOY_CPU(LogGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LogGradient);
#endif

OPERATOR_SCHEMA(Log)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(LogGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Log, SimpleGradientMaker);

}  // namespace dragon