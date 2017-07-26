#include "operators/activation/sigmoid_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::Sigmoid<T, Context>(output(0)->count(), Xdata, Ydata);
}

template <class Context>
void SigmoidOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Sigmoid);
#ifdef WITH_CUDA
DEPLOY_CUDA(Sigmoid);
#endif
OPERATOR_SCHEMA(Sigmoid).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void SigmoidGradientOp<Context>::RunWithType() {
    auto* Ydata = input(0).template data<T, Context>();
    auto* dYdata = input(1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    kernel::SigmoidGrad<T, Context>(output(0)->count(), dYdata, Ydata, dXdata);
}

template <class Context>
void SigmoidGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SigmoidGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidGradient);
#endif
OPERATOR_SCHEMA(SigmoidGradient).NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 } });

class GetSigmoidGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetSigmoidGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Sigmoid, GetSigmoidGradient);

}    // namespace dragon