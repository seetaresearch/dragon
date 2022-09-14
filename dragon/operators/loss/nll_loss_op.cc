#include "dragon/operators/loss/nll_loss_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void NLLLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  CHECK_EQ(Y.count(), NxS) << "\nNumel of X and Y must be matched.";

  auto* input = X.template mutable_data<T, Context>();
  auto* scratch = ctx()->workspace()->template data<T, Context>(NxS * 2 + 1);
  auto *loss = scratch, *mask = scratch + NxS;

  if (Y.meta() == TypeMeta::Make<int>()) {
    auto* target = Y.template data<int, Context>();
    kernels::NLLLoss(N, S, C, ignore_index_, input, target, loss, mask, ctx());
  } else if (Y.meta() == TypeMeta::Make<int64_t>()) {
    auto* target = Y.template data<int64_t, Context>();
    kernels::NLLLoss(N, S, C, ignore_index_, input, target, loss, mask, ctx());
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtypes::to_string(Y.meta()), {"int32", "int64"});
  }

  if (reduction_ == "NONE") {
    auto out_dims = X.dims();
    out_dims.erase(out_dims.begin() + axis);
    math::Copy(
        NxS,
        loss,
        L->Reshape(out_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1;
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = NxS;
    }
    kernels::ReduceLoss(
        NxS,
        NxS,
        normalizer,
        loss,
        mask,
        L->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void NLLLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dL = Input(2);
  auto* dX = Output(0)->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  const auto NxCxS = X.count();

  auto* input = X.template data<T, Context>();
  auto* dl = dL.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  auto* mask = ctx()->workspace()->template data<T, Context>(NxS + 1);
  math::Set(dX->count(), convert::To<T>(0.f), dx, ctx());

  if (Y.meta() == TypeMeta::Make<int>()) {
    auto* target = Y.template data<int, Context>();
    kernels::NLLLossGrad(
        N, S, C, ignore_index_, input, target, dx, mask, ctx());
  } else if (Y.meta() == TypeMeta::Make<int64_t>()) {
    auto* target = Y.template data<int64_t, Context>();
    kernels::NLLLossGrad(
        N, S, C, ignore_index_, input, target, dx, mask, ctx());
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtypes::to_string(Y.meta()), {"int32", "int64"});
  }

  if (reduction_ == "NONE") {
    kernels::BroadcastLossGrad(N, S, C, dl, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1;
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = NxS;
    }
    kernels::ReduceLossGrad(NxCxS, NxS, normalizer, dl, mask, dx, ctx());
  }
}

DEPLOY_CPU_OPERATOR(NLLLoss);
DEPLOY_CPU_OPERATOR(NLLLossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(NLLLoss);
DEPLOY_CUDA_OPERATOR(NLLLossGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(NLLLoss, NLLLoss);
DEPLOY_MPS_OPERATOR(NLLLossGradient, NLLLossGradient);
#endif

OPERATOR_SCHEMA(NLLLoss)
    /* X, T */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(NLLLossGradient)
    /* X, T, dY */
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

REGISTER_GRADIENT(NLLLoss, GradientMaker);

} // namespace dragon
