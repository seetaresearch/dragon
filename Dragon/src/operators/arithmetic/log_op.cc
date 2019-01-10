#include "utils/math_functions.h"
#include "operators/arithmetic/log_op.h"

namespace dragon {

template <class Context> template <typename T>
void LogOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Log(Output(0)->count(), Xdata, Ydata, ctx());
}

template <class Context>
void LogOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), { 
        "float16", "float32", "float64"});
}

DEPLOY_CPU(Log);
#ifdef WITH_CUDA
DEPLOY_CUDA(Log);
#endif
OPERATOR_SCHEMA(Log).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void LogGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    math::Div(Output(0)->count(), dYdata, Xdata, dXdata, ctx());
}

template <class Context>
void LogGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(LogGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LogGradient);
#endif

OPERATOR_SCHEMA(LogGradient)
    .NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Log, SimpleGradientMaker);

}  // namespace dragon