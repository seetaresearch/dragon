#include "dragon/core/workspace.h"
#include "dragon/operators/loss/softmax_loss_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename LogitType, typename TargetType>
void SparseSoftmaxCrossEntropyOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  auto outer_dim = X.count(0, axis);
  auto inner_dim = X.count(axis + 1);
  auto num_preds = outer_dim * inner_dim;

  CHECK_EQ(num_preds, Input(1).count())
      << "\nNumber of preds must match the number of targets.";
  auto* X_prob = Buffer("prob")->ReshapeLike(X);
  auto* prob = X_prob->template mutable_data<LogitType, Context>();

  auto scratches = ctx()->workspace()->template data<Context>({
      (size_t)num_preds * sizeof(LogitType), // loss
      (size_t)num_preds * sizeof(LogitType) + sizeof(LogitType), // mask
  });
  auto* loss = static_cast<LogitType*>(scratches[0]);
  auto* mask = static_cast<LogitType*>(scratches[1]);

  kernel::Softmax(
      outer_dim,
      inner_dim,
      X.dim(axis),
      X.template data<LogitType, Context>(),
      prob,
      ctx());

  kernel::SparseSoftmaxCrossEntropy(
      outer_dim,
      inner_dim,
      X.dim(axis),
      ignore_index_,
      prob,
      Input(1).template data<TargetType, Context>(),
      loss,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    auto out_shape = X.dims();
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
    } else if (reduction_ == "BATCH_MEAN") {
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
void SparseSoftmaxCrossEntropyOp<Context>::RunOnDevice() {
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
void SparseSoftmaxCrossEntropyGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(-1), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(Input(0));

  auto outer_dim = dX->count(0, axis);
  auto inner_dim = dX->count(axis + 1);
  auto num_preds = outer_dim * inner_dim;

  auto* prob = Buffer("prob")->template data<LogitType, Context>();
  auto* dy = Input(-1).template data<LogitType, Context>();
  auto* dx = Output(0)->template mutable_data<LogitType, Context>();
  auto* mask =
      ctx()->workspace()->template data<LogitType, Context>({num_preds + 1})[0];

  math::Copy(dX->count(), prob, dx, ctx());

  kernel::SparseSoftmaxCrossEntropyGrad(
      outer_dim,
      inner_dim,
      dX->dim(axis),
      ignore_index_,
      prob,
      Input(1).template data<TargetType, Context>(),
      dx,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    kernel::BroadcastLossGrad(
        outer_dim, inner_dim, dX->dim(axis), dy, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = -1; // Select from mask
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = dX->dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = num_preds;
    }
    kernel::ReduceLossGrad(
        dX->count(), num_preds, normalizer, dy, mask, dx, ctx());
  }
}

template <class Context>
void SparseSoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));

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

DEPLOY_CPU_OPERATOR(SparseSoftmaxCrossEntropy);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SparseSoftmaxCrossEntropy);
#endif

DEPLOY_CPU_OPERATOR(SparseSoftmaxCrossEntropyGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SparseSoftmaxCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SparseSoftmaxCrossEntropy)
    /* X, T */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SparseSoftmaxCrossEntropyGradient)
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

REGISTER_GRADIENT(SparseSoftmaxCrossEntropy, GradientMaker);

} // namespace dragon
