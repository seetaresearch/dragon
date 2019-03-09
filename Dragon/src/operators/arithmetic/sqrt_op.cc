#include "utils/math_functions.h"
#include "operators/arithmetic/sqrt_op.h"

namespace dragon {

template <class Context> template <typename T>
void SqrtOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Sqrt(Output(0)->count(), Xdata, Ydata, ctx());
}

template <class Context>
void SqrtOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0),
        {"float16", "float32", "float64"});
}

DEPLOY_CPU(Sqrt);
#ifdef WITH_CUDA
DEPLOY_CUDA(Sqrt);
#endif
OPERATOR_SCHEMA(Sqrt)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void SqrtGradientOp<Context>::RunWithType() {
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dYdata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    math::Scale(Output(0)->count(), 2.f, Ydata, dXdata, ctx());
    math::Inv(Output(0)->count(), dXdata, dXdata, ctx());
}

template <class Context>
void SqrtGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0),
        {"float16", "float32", "float64" });
}

DEPLOY_CPU(SqrtGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SqrtGradient);
#endif

OPERATOR_SCHEMA(SqrtGradient)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Sqrt, InplaceGradientMaker);

}  // namespace dragon