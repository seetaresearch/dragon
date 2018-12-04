#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/loss/ctc_loss_op.h"

namespace dragon {

DEPLOY_CPU(CTCLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(CTCLoss);
#endif
OPERATOR_SCHEMA(CTCLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void CTCLossGradientOp<Context>::RunWithType() {
    auto* gradT = ws()->GetTensor(
        "/mnt/" + anchor() + "/ctc/grads");
    Output(0)->ReshapeLike(*gradT);

    auto* Gdata = gradT->template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();

    T dYdata_host; ctx()->template Copy<T, CPUContext, Context>(
        1, &dYdata_host, dYdata);
    math::Scale<T, Context>(Output(0)->count(),
        dYdata_host, Gdata, dXdata, ctx());
}

template <class Context>
void CTCLossGradientOp<Context>::RunOnDevice() {
    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(CTCLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(CTCLossGradient);
#endif
OPERATOR_SCHEMA(CTCLossGradient).NumInputs(1).NumOutputs(1);

class GetCTCLossGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetCTCLossGradient);
    vector<OperatorDef> MakeDefs() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(CTCLoss, GetCTCLossGradient);

}  // namespace dragon