#include "dragon/operators/loss/nll_loss_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename LogitType, typename TargetType>
void NLLLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  auto outer_dim = X.count(0, axis);
  auto inner_dim = X.count(axis + 1);
  auto num_preds = outer_dim * inner_dim;

  CHECK_EQ(num_preds, Input(1).count())
      << "\nNumber of preds must match the number of targets.";

  auto scratches = ws()->template data<Context>({
      num_preds * sizeof(LogitType), // loss
      num_preds * sizeof(int), // mask
  });
  auto* loss = static_cast<LogitType*>(scratches[0]);
  auto* mask = static_cast<int*>(scratches[1]);

  kernel::NLLLoss(
      outer_dim,
      X.dim(axis),
      inner_dim,
      ignore_index_,
      X.template data<LogitType, Context>(),
      Input(1).template data<TargetType, Context>(),
      loss,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    auto out_shape = Input(0).dims();
    out_shape.erase(out_shape.begin() + axis);
    math::Copy(
        num_preds,
        loss,
        Y->Reshape(out_shape)->template mutable_data<LogitType, Context>(),
        ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_SIZE") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = num_preds;
    }
    kernel::ReduceLoss(
        num_preds,
        num_preds,
        normalizer,
        loss,
        mask,
        Y->Reshape({})->template mutable_data<LogitType, Context>(),
        ctx());
  }
}

template <class Context>
void NLLLossOp<Context>::RunOnDevice() {
  if (Input(0).template IsType<float>()) {
    if (Input(1).template IsType<float>()) {
      DoRunWithType<float, float>();
    } else if (Input(1).template IsType<int64_t>()) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float32", "int64"});
    }
  } else if (Input(0).template IsType<double>()) {
    if (Input(1).template IsType<double>()) {
      DoRunWithType<double, double>();
    } else if (Input(1).template IsType<int64_t>()) {
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
void NLLLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(-1), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);
  dX->ReshapeLike(X);

  auto outer_dim = dX->count(0, axis);
  auto inner_dim = dX->count(axis + 1);
  auto num_preds = outer_dim * inner_dim;

  auto* mask = ws()->template data<int, Context>({num_preds})[0];
  auto* dy = dY.template data<LogitType, Context>();
  auto* dx = dX->template mutable_data<LogitType, Context>();
  math::Set(dX->count(), cast::to<LogitType>(0.f), dx, ctx());

  kernel::NLLLossGrad(
      outer_dim,
      dX->dim(axis),
      inner_dim,
      ignore_index_,
      X.template data<LogitType, Context>(),
      Input(1).template data<TargetType, Context>(),
      dx,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    kernel::BroadcastLossGrad(
        outer_dim, dX->dim(axis), inner_dim, dy, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_SIZE") {
      normalizer = dX->dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = num_preds;
    }
    kernel::ReduceLossGrad(
        dX->count(), num_preds, normalizer, dy, mask, dx, ctx());
  }
}

template <class Context>
void NLLLossGradientOp<Context>::RunOnDevice() {
  if (Input(0).template IsType<float>()) {
    if (Input(1).template IsType<float>()) {
      DoRunWithType<float, float>();
    } else if (Input(1).template IsType<int64_t>()) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float32", "int64"});
    }
  } else if (Input(0).template IsType<double>()) {
    if (Input(1).template IsType<double>()) {
      DoRunWithType<double, double>();
    } else if (Input(1).template IsType<int64_t>()) {
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

DEPLOY_CPU_OPERATOR(NLLLoss);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(NLLLoss);
#endif

DEPLOY_CPU_OPERATOR(NLLLossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(NLLLossGradient);
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
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(NLLLoss, GradientMaker);

} // namespace dragon
