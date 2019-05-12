#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/tanh_op.h"

namespace dragon {

template <class Context> template <typename T>
void TanhOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    kernel::Tanh(X(0).count(), x, y, ctx());
}

template <class Context>
void TanhOp<Context>::RunOnDevice() {
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
void TanhGradientOp<Context>::RunImpl() {
    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::TanhGrad(
        X(0).count(),
        dy, y,
        dx, ctx()
    );
}

template <class Context>
void TanhGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(Tanh);
#ifdef WITH_CUDA
DEPLOY_CUDA(Tanh);
#endif

DEPLOY_CPU(TanhGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TanhGradient);
#endif

OPERATOR_SCHEMA(Tanh)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(TanhGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Tanh, InplaceGradientMaker);

}  // namespace dragon