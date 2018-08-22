#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/sigmoid_op.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::Sigmoid<T, Context>(Output(0)->count(), Xdata, Ydata, ctx());
}

template <class Context>
void SigmoidOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Sigmoid);
#ifdef WITH_CUDA
DEPLOY_CUDA(Sigmoid);
#endif
OPERATOR_SCHEMA(Sigmoid).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void SigmoidGradientOp<Context>::RunWithType() {
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dYdata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    kernel::SigmoidGrad<T, Context>(Output(0)->count(),
        dYdata, Ydata, dXdata, ctx());
}

template <class Context>
void SigmoidGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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