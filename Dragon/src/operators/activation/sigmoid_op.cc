#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/sigmoid_op.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    kernel::Sigmoid(X(0).count(), x, y, ctx());
}

template <class Context>
void SigmoidOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void SigmoidGradientOp<Context>::RunImpl() {
    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::SigmoidGrad(
        X(0).count(),
        dy, y,
        dx, ctx()
    );
}

template <class Context>
void SigmoidGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(Sigmoid);
#ifdef WITH_CUDA
DEPLOY_CUDA(Sigmoid);
#endif

DEPLOY_CPU(SigmoidGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidGradient);
#endif

OPERATOR_SCHEMA(Sigmoid)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(SigmoidGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Sigmoid, InplaceGradientMaker);

}  // namespace dragon