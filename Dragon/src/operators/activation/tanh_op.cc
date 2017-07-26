#include "operators/activation/tanh_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void TanhOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::Tanh<T, Context>(output(0)->count(), Xdata, Ydata);
}

template <class Context>
void TanhOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Tanh);
#ifdef WITH_CUDA
DEPLOY_CUDA(Tanh);
#endif
OPERATOR_SCHEMA(Tanh).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void TanhGradientOp<Context>::RunWithType() {
    auto* Ydata = input(0).template data<T, Context>();
    auto* dYdata = input(1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    kernel::TanhGrad<T, Context>(output(0)->count(), dYdata, Ydata, dXdata);
}

template <class Context>
void TanhGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(TanhGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TanhGradient);
#endif
OPERATOR_SCHEMA(TanhGradient).NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 } });

class GetTanhGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetTanhGradient);
    vector<OperatorDef>    MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Tanh, GetTanhGradient);

}    // namespace dragon

