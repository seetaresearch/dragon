#include "dragon/core/workspace.h"
#include "dragon/operators/loss/cross_entropy_loss_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SigmoidCrossEntropyLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);

  const auto N = X.count();
  CHECK_EQ(Y.count(), N) << "\nNumel of X and Y must be matched.";

  auto* scratch = ctx()->workspace()->template data<T, Context>(N * 2 + 1);
  auto *loss = scratch, *mask = scratch + N;

  kernels::SigmoidCrossEntropy(
      N,
      X.template data<T, Context>(),
      Y.template data<T, Context>(),
      loss,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    math::Copy(
        N, loss, L->ReshapeLike(X)->template mutable_data<T, Context>(), ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = N;
    }
    kernels::ReduceLoss(
        N,
        N,
        normalizer,
        loss,
        mask,
        L->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void SigmoidCrossEntropyLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dL = Input(2);
  auto* dX = Output(0)->ReshapeLike(X);

  const auto N = X.count();
  auto* dl = dL.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  auto* mask = ctx()->workspace()->template data<T, Context>(N + 1);

  kernels::SigmoidCrossEntropyGrad(
      N,
      X.template data<T, Context>(),
      Y.template data<T, Context>(),
      dx,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    math::Mul(N, dl, dx, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = N;
    }
    kernels::ReduceLossGrad(N, N, normalizer, dl, mask, dx, ctx());
  }
}

DEPLOY_CPU_OPERATOR(SigmoidCrossEntropyLoss);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SigmoidCrossEntropyLoss);
#endif

DEPLOY_CPU_OPERATOR(SigmoidCrossEntropyLossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SigmoidCrossEntropyLossGradient);
#endif

OPERATOR_SCHEMA(SigmoidCrossEntropyLoss)
    /* X, Y */
    .NumInputs(2)
    /* L */
    .NumOutputs(1);

OPERATOR_SCHEMA(SigmoidCrossEntropyLossGradient)
    /* X, Y, dL */
    .NumInputs(3)
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
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(SigmoidCrossEntropyLoss, GradientMaker);

} // namespace dragon
