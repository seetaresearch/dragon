#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/selu_op.h"

namespace dragon {

template <class Context> template <typename T>
void SEluOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::SElu(
        X(0).count(),
        x, y, ctx()
    );
}

template <class Context>
void SEluOp<Context>::RunOnDevice() {
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
void SEluGradientOp<Context>::RunImpl() {
    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::SEluGrad(
        X(0).count(),
        dy, y,
        dx, ctx()
    );
}

template <class Context>
void SEluGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(SElu);
#ifdef WITH_CUDA
DEPLOY_CUDA(SElu);
#endif

DEPLOY_CPU(SEluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SEluGradient);
#endif

OPERATOR_SCHEMA(SElu)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(SEluGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 }});

REGISTER_GRADIENT(SElu, InplaceGradientMaker);

}  // namespace dragon