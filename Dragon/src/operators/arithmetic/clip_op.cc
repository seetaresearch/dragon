#include "operators/arithmetic/clip_op.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "core/workspace.h"

namespace dragon {

template <class Context> template <typename T>
void ClipOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template mutable_data<T, Context>();
    kernel::Clip<T, Context>(output(0)->count(), low, high, Xdata, Mdata, Ydata);
}

template <class Context>
void ClipOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    output(0)->Share(input(0));
    mask = ws()->CreateTensor("_t_" + anchor() + "_clip_mask");
    mask->ReshapeLike(input(0));
    if (input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Clip);
#ifdef WITH_CUDA
DEPLOY_CUDA(Clip);
#endif
OPERATOR_SCHEMA(Clip).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void ClipGradientOp<Context>::RunWithType() {
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template data<T, Context>();
    math::Mul<T, Context>(output(0)->count(), dXdata, Mdata, dXdata);
}

template <class Context>
void ClipGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    output(0)->Share(input(-1));
    mask = ws()->GetTensor("_t_" + anchor() + "_clip_mask");
    if (input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
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