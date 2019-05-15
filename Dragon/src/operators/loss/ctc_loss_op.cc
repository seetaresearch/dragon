#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/loss/ctc_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void CTCLossGradientOp<Context>::RunImpl() {
    auto* G = ws()->GetTensor(unique_name("grad"));
    Y(0)->ReshapeLike(*G);

    auto* g  = G->template data<T, Context>();
    auto* dy = X(0).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    T dyHost; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dyHost, dy);
    ctx()->FinishDeviceCompution();

    math::Scale(Y(0)->count(), dyHost, g, dx, ctx());
}

template <class Context>
void CTCLossGradientOp<Context>::RunOnDevice() {
    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(CTCLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(CTCLoss);
#endif

DEPLOY_CPU(CTCLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(CTCLossGradient);
#endif

OPERATOR_SCHEMA(CTCLoss)
     /* X, T */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(CTCLossGradient)
     /* dY */
    .NumInputs(1)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ GO(0) }),
            vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(CTCLoss, GradientMaker);

}  // namespace dragon