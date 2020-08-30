#include "dragon/operators/loss/ctc_loss_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CTCLossGradientOp<Context>::DoRunWithType() {
  auto* G = Buffer("grad");
  Output(0)->ReshapeLike(*G);

  auto* g = G->template data<T, Context>();
  auto* dy = Input(0).template data<T, Context>();
  auto* dx = Output(0)->template mutable_data<T, Context>();

  T alpha;
  ctx()->template Copy<T, CPUContext, Context>(1, &alpha, dy);
  ctx()->FinishDeviceComputation();

  math::Scale(Output(0)->count(), alpha, g, dx, ctx());
}

template <class Context>
void CTCLossGradientOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float>>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(CTCLoss);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(CTCLoss);
#endif

DEPLOY_CPU_OPERATOR(CTCLossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(CTCLossGradient);
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
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(CTCLoss, GradientMaker);

} // namespace dragon
