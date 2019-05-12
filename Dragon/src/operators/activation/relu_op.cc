#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/relu_op.h"

namespace dragon {

template <class Context> template <typename T>
void ReluOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Relu(
        X(0).count(),
        slope_, x,
        y, ctx()
    );
}

template <class Context>
void ReluOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template <class Context> template <typename T>
void ReluGradientOp<Context>::RunImpl() {
    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::ReluGrad(
        X(0).count(),
        slope_, dy, y,
        dx, ctx()
    );
}

template <class Context>
void ReluGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CPU(Relu);
#ifdef WITH_CUDA
DEPLOY_CUDA(Relu);
#endif

DEPLOY_CPU(ReluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReluGradient);
#endif

OPERATOR_SCHEMA(Relu)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(ReluGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 }});

REGISTER_GRADIENT(Relu, InplaceGradientMaker);

}  // namespace dragon