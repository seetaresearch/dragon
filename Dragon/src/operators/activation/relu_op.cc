#include "operators/activation/relu_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ReluOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::Relu<T, Context>(Output(0)->count(), slope, Xdata, Ydata);
}

template <class Context>
void ReluOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Relu);
#ifdef WITH_CUDA
DEPLOY_CUDA(Relu);
#endif
OPERATOR_SCHEMA(Relu).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void ReluGradientOp<Context>::RunWithType() {
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dYdata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    kernel::ReluGrad<T, Context>(
        Output(0)->count(), slope, dYdata, Ydata, dXdata);
}

template <class Context>
void ReluGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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