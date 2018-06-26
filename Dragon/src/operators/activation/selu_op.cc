#include "operators/activation/selu_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SEluOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::SElu<T, Context>(Output(0)->count(), Xdata, Ydata);
}

template <class Context>
void SEluOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SElu);
#ifdef WITH_CUDA
DEPLOY_CUDA(SElu);
#endif
OPERATOR_SCHEMA(SElu).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void SEluGradientOp<Context>::RunWithType() {
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dYdata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    kernel::SEluGrad<T, Context>(
        Output(0)->count(), dYdata, Ydata, dXdata);
}

template <class Context>
void SEluGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SEluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SEluGradient);
#endif
OPERATOR_SCHEMA(SEluGradient).NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 }});

class GetSEluGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSEluGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(SElu, GetSEluGradient);

}    // namespace dragon