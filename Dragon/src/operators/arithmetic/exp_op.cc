#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/exp_op.h"

namespace dragon {

template <class Context> template <typename T>
void ExpOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Exp(Output(0)->count(), Xdata, Ydata, ctx());
}

template <class Context>
void ExpOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(Exp);
#ifdef WITH_CUDA
DEPLOY_CUDA(Exp);
#endif
OPERATOR_SCHEMA(Exp).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void ExpGradientOp<Context>::RunWithType() {
    auto* Ydata = Input(0).template data<T, Context >();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    math::Mul(Output(0)->count(), dYdata, Ydata, dXdata, ctx());
}

template <class Context>
void ExpGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(ExpGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpGradient);
#endif

OPERATOR_SCHEMA(ExpGradient)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Exp, InplaceGradientMaker);

}  // namespace dragon