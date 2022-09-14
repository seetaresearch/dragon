#include "dragon/operators/loss/ctc_loss_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void CTCLossGradientOp<Context>::DoRunWithType() {
  auto &dL = Input(0), &X_grad = Input("X_grad");
  auto* dX = Output(0)->ReshapeLike(X_grad)->CopyFrom(X_grad, ctx());
  kernels::ReduceLossGrad(
      dX->count(),
      0,
      1.f,
      dL.template data<T, Context>(),
      (T*)nullptr,
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void CTCLossGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::TypesBase<float>>::Call(this, Input(0));
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
    /* X, Y */
    .NumInputs(2)
    /* L */
    .NumOutputs(1);

OPERATOR_SCHEMA(CTCLossGradient)
    /* dL */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(CTCLoss, GradientMaker);

} // namespace dragon
