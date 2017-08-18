#include "operators/activation/relu_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ReluOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::Relu<T, Context>(output(0)->count(), Xdata, slope, Ydata);
}

template <class Context>
void ReluOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Relu);
#ifdef WITH_CUDA
DEPLOY_CUDA(Relu);
#endif
OPERATOR_SCHEMA(Relu).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void ReluGradientOp<Context>::RunWithType() {
    auto* Ydata = input(0).template data<T, Context>();
    auto* dYdata = input(1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    kernel::ReluGrad<T, Context>(output(0)->count(), dYdata, Ydata, slope, dXdata);
}

template <class Context>
void ReluGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(ReluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReluGradient);
#endif
OPERATOR_SCHEMA(ReluGradient).NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 }});

class GetReluGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetReluGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Relu, GetReluGradient);

}    // namespace dragon