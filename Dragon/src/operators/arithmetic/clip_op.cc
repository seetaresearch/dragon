#include "operators/arithmetic/clip_op.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "core/workspace.h"

namespace dragon {

template <class Context> template <typename T>
void ClipOp<Context>::RunWithType() {
    Tensor* mask = ws()->CreateTensor(
        "/mnt/" + anchor() + "/clip/mask");
    mask->ReshapeLike(Input(0));

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template mutable_data<T, Context>();
    kernel::Clip<T, Context>(Output(0)->count(),
        low, high, Xdata, Mdata, Ydata);
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
OPERATOR_SCHEMA(Clip).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void ClipGradientOp<Context>::RunWithType() {
    Tensor* mask = ws()->GetTensor(
        "/mnt/" + anchor() + "/clip/mask");

    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(), dXdata, Mdata, dXdata);
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
OPERATOR_SCHEMA(ClipGradient).NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 } });

class GetClipGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetClipGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Clip, GetClipGradient);

}   // namespace dragon