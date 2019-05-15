#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/elu_op.h"

namespace dragon {

template <class Context> template <typename T>
void EluOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Elu(
        X(0).count(),
        alpha_, x,
        y, ctx()
    );
}

template <class Context>
void EluOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void EluGradientOp<Context>::RunImpl() {
    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::EluGrad(
        X(0).count(),
        alpha_, dy, y,
        dx, ctx()
    );
}

template <class Context>
void EluGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(Elu);
#ifdef WITH_CUDA
DEPLOY_CUDA(Elu);
#endif

DEPLOY_CPU(EluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(EluGradient);
#endif

OPERATOR_SCHEMA(Elu)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(EluGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 }});

REGISTER_GRADIENT(Elu, InplaceGradientMaker);

}  // namespace dragon