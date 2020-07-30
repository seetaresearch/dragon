#include "dragon/core/workspace.h"
#include "dragon/operators/loss/sigmoid_loss_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename LogitType, typename TargetType>
void SigmoidFocalLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  auto outer_dim = X.count(0, axis);
  auto inner_dim = X.count(axis + 1);

  CHECK_EQ(outer_dim * inner_dim, Input(1).count())
      << "\nNumber of preds must match the number of targets.";

  auto scratches = ws()->template data<Context>({
      X.count() * sizeof(LogitType), // loss
      X.count() * sizeof(int), // mask
  });
  auto* loss = static_cast<LogitType*>(scratches[0]);
  auto* mask = static_cast<int*>(scratches[1]);

  kernel::SigmoidFocalLoss(
      outer_dim,
      X.dim(axis),
      inner_dim,
      pos_alpha_,
      neg_alpha_,
      gamma_,
      negative_index_,
      X.template data<LogitType, Context>(),
      Input(1).template data<TargetType, Context>(),
      loss,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    math::Copy(
        X.count(),
        loss,
        Y->ReshapeLike(X)->template mutable_data<LogitType, Context>(),
        ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_SIZE") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = X.count();
    }
    kernel::ReduceLoss(
        X.count(),
        X.count(),
        normalizer,
        loss,
        mask,
        Y->Reshape({})->template mutable_data<LogitType, Context>(),
        ctx());
  }
}

template <class Context>
void SigmoidFocalLossOp<Context>::RunOnDevice() {
  if (XIsType(Input(0), float)) {
    if (XIsType(Input(1), float)) {
      DoRunWithType<float, float>();
    } else if (XIsType(Input(1), int64_t)) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float32", "int64"});
    }
  } else if (XIsType(Input(0), double)) {
    if (XIsType(Input(1), double)) {
      DoRunWithType<double, double>();
    } else if (XIsType(Input(1), int64_t)) {
      DoRunWithType<double, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float64", "int64"});
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float32", "float64"});
  }
}

template <class Context>
template <typename LogitType, typename TargetType>
void SigmoidFocalLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(-1), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);
  dX->ReshapeLike(X);

  auto outer_dim = dX->count(0, axis);
  auto inner_dim = dX->count(axis + 1);

  auto* mask = ws()->template data<int, Context>({dX->count()})[0];
  auto* dy = dY.template data<LogitType, Context>();
  auto* dx = dX->template mutable_data<LogitType, Context>();

  kernel::SigmoidFocalLossGrad(
      outer_dim,
      dX->dim(axis),
      inner_dim,
      pos_alpha_,
      neg_alpha_,
      gamma_,
      negative_index_,
      X.template data<LogitType, Context>(),
      Input(1).template data<TargetType, Context>(),
      dx,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    math::Mul(dX->count(), dy, dx, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_SIZE") {
      normalizer = dX->dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = dX->count();
    }
    kernel::ReduceLossGrad(
        dX->count(), dX->count(), normalizer, dy, mask, dx, ctx());
  }
}

template <class Context>
void SigmoidFocalLossGradientOp<Context>::RunOnDevice() {
  if (XIsType(Input(0), float)) {
    if (XIsType(Input(1), float)) {
      DoRunWithType<float, float>();
    } else if (XIsType(Input(1), int64_t)) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float32", "int64"});
    }
  } else if (XIsType(Input(0), double)) {
    if (XIsType(Input(1), double)) {
      DoRunWithType<double, double>();
    } else if (XIsType(Input(1), int64_t)) {
      DoRunWithType<double, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float64", "int64"});
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float32", "float64"});
  }
}

DEPLOY_CPU(SigmoidFocalLoss);
#ifdef USE_CUDA
DEPLOY_CUDA(SigmoidFocalLoss);
#endif

DEPLOY_CPU(SigmoidFocalLossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(SigmoidFocalLossGradient);
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
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(SigmoidFocalLoss, GradientMaker);

} // namespace dragon
