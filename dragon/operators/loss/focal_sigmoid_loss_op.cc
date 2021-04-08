#include "dragon/core/workspace.h"
#include "dragon/operators/loss/focal_loss_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename InputT, typename TargetT>
void SigmoidFocalLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  const auto NxCxS = X.count();
  CHECK_EQ(Y.count(), NxS) << "\nNumel of X and Y must be matched.";

  auto scratches = ctx()->workspace()->template data<Context>({
      size_t(NxCxS) * sizeof(InputT),
      size_t(NxCxS) * sizeof(InputT) + sizeof(InputT),
  });
  auto* loss = static_cast<InputT*>(scratches[0]);
  auto* mask = static_cast<InputT*>(scratches[1]);

  kernels::SigmoidFocalLoss(
      N,
      S,
      C,
      start_index_,
      alpha_,
      gamma_,
      X.template data<InputT, Context>(),
      Y.template data<TargetT, Context>(),
      loss,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    math::Copy(
        NxCxS,
        loss,
        L->ReshapeLike(X)->template mutable_data<InputT, Context>(),
        ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = NxCxS;
    }
    kernels::ReduceLoss(
        NxCxS,
        NxCxS,
        normalizer,
        loss,
        mask,
        L->Reshape({})->template mutable_data<InputT, Context>(),
        ctx());
  }
}

template <class Context>
void SigmoidFocalLossOp<Context>::RunOnDevice() {
  auto &X = Input(0), &Y = Input(1);
  if (X.template IsType<float>()) {
    if (Y.template IsType<int>()) {
      DoRunWithType<float, int>();
    } else if (Y.template IsType<int64_t>()) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(Y.meta()), {"int32", "int64"});
    }
  } else if (X.template IsType<double>()) {
    if (Y.template IsType<int>()) {
      DoRunWithType<double, int>();
    } else if (Y.template IsType<int64_t>()) {
      DoRunWithType<double, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(Y.meta()), {"int32", "int64"});
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtypes::to_string(X.meta()), {"float32", "float64"});
  }
}

template <class Context>
template <typename InputT, typename TargetT>
void SigmoidFocalLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dL = Input(2);
  auto* dX = Output(0)->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  const auto NxCxS = X.count();

  auto* dl = dL.template data<InputT, Context>();
  auto* dx = dX->template mutable_data<InputT, Context>();
  auto* mask =
      ctx()->workspace()->template data<InputT, Context>({NxCxS + 1})[0];

  kernels::SigmoidFocalLossGrad(
      N,
      S,
      C,
      start_index_,
      alpha_,
      gamma_,
      X.template data<InputT, Context>(),
      Y.template data<TargetT, Context>(),
      dx,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    math::Mul(NxCxS, dl, dx, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = NxCxS;
    }
    kernels::ReduceLossGrad(NxCxS, NxCxS, normalizer, dl, mask, dx, ctx());
  }
}

template <class Context>
void SigmoidFocalLossGradientOp<Context>::RunOnDevice() {
  auto &X = Input(0), &Y = Input(1);
  if (X.template IsType<float>()) {
    if (Y.template IsType<int>()) {
      DoRunWithType<float, int>();
    } else if (Y.template IsType<int64_t>()) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(Y.meta()), {"int32", "int64"});
    }
  } else if (X.template IsType<double>()) {
    if (Y.template IsType<int>()) {
      DoRunWithType<double, int>();
    } else if (Y.template IsType<int64_t>()) {
      DoRunWithType<double, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(Y.meta()), {"int32", "int64"});
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtypes::to_string(X.meta()), {"float32", "float64"});
  }
}

DEPLOY_CPU_OPERATOR(SigmoidFocalLoss);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SigmoidFocalLoss);
#endif

DEPLOY_CPU_OPERATOR(SigmoidFocalLossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SigmoidFocalLossGradient);
#endif

OPERATOR_SCHEMA(SigmoidFocalLoss)
    /* X, T */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SigmoidFocalLossGradient)
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

REGISTER_GRADIENT(SigmoidFocalLoss, GradientMaker);

} // namespace dragon
