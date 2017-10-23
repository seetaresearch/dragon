#include "operators/arithmetic/log_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void LogOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    math::Log<T, Context>(output(0)->count(), Xdata, Ydata);
}

template <class Context>
void LogOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(Log);
#ifdef WITH_CUDA
DEPLOY_CUDA(Log);
#endif
OPERATOR_SCHEMA(Log).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void LogGradientOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    math::Div<T, Context>(output(0)->count(), dYdata, Xdata, dXdata);
}

template <class Context>
void LogGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(LogGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LogGradient);
#endif
OPERATOR_SCHEMA(LogGradient).NumInputs(2).NumOutputs(1);

class GetLogGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetLogGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Log, GetLogGradient);

}    // namespace dragon