#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/clip_op.h"

namespace dragon {

template <class Context> template <typename T>
void ClipOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Clip<T, Context>(Output(0)->count(),
        low, high, Xdata, Ydata, ctx());
}

template <class Context>
void ClipOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Clip);
#ifdef WITH_CUDA
DEPLOY_CUDA(Clip);
#endif
OPERATOR_SCHEMA(Clip).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void ClipGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::ClipGrad<T, Context>(Output(0)->count(),
        low, high, Xdata, dYdata, dXdata, ctx());
}

template <class Context>
void ClipGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(ClipGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ClipGradient);
#endif
OPERATOR_SCHEMA(ClipGradient).NumInputs(2).NumOutputs(1);

class GetClipGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetClipGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Clip, GetClipGradient);

}   // namespace dragon