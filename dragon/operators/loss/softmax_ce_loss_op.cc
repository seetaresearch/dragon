#include "dragon/core/workspace.h"
#include "dragon/operators/loss/softmax_loss_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SoftmaxCrossEntropyOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  auto outer_dim = X.count(0, axis);
  auto inner_dim = X.count(axis + 1);
  auto num_preds = outer_dim * inner_dim;

  CHECK_EQ(X.count(), Input(1).count())
      << "\nNumber of preds must match the number of targets.";
  Buffer("prob")->ReshapeLike(X);

  auto* loss = ws()->template data<T, Context>({X.count()})[0];
  auto* prob = Buffer("prob")->template mutable_data<T, Context>();

  kernel::Softmax(
      outer_dim,
      X.dim(axis),
      inner_dim,
      X.template data<T, Context>(),
      prob,
      ctx());

  kernel::SoftmaxCrossEntropy(
      X.count(), prob, Input(1).template data<T, Context>(), loss, ctx());

  if (reduction_ == "NONE") {
    auto Y_dims = X.dims();
    Y_dims.erase(Y_dims.begin() + axis);
    vec32_t dims = {(int)outer_dim, (int)X.dim(axis), (int)inner_dim};
    vec32_t axes = {1};
    math::ReduceSum(
        3,
        dims.data(),
        1,
        axes.data(),
        1.f,
        loss,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_SIZE") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = num_preds;
    }
    kernel::ReduceLoss(
        X.count(),
        0,
        normalizer,
        loss,
        nullptr,
        Y->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void SoftmaxCrossEntropyOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SoftmaxCrossEntropyGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(-1), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(Input(0));

  auto outer_dim = dX->count(0, axis);
  auto inner_dim = dX->count(axis + 1);
  auto num_preds = outer_dim * inner_dim;

  auto* prob = Buffer("prob")->template data<T, Context>();
  auto* target = Input(1).template data<T, Context>();
  auto* dy = Input(-1).template data<T, Context>();
  auto* dx = Output(0)->template mutable_data<T, Context>();

  math::Copy(dX->count(), prob, dx, ctx());
  math::Axpy(dX->count(), -1.f, target, dx, ctx());

  if (reduction_ == "NONE") {
    kernel::BroadcastLossGrad(
        outer_dim, dX->dim(axis), inner_dim, dy, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_SIZE") {
      normalizer = dX->dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = num_preds;
    }
    kernel::ReduceLossGrad(dX->count(), 0, normalizer, dy, nullptr, dx, ctx());
  }
}

template <class Context>
void SoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(SoftmaxCrossEntropy);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SoftmaxCrossEntropy);
#endif

DEPLOY_CPU_OPERATOR(SoftmaxCrossEntropyGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SoftmaxCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SoftmaxCrossEntropy)
    /* X, T */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SoftmaxCrossEntropyGradient)
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

REGISTER_GRADIENT(SoftmaxCrossEntropy, GradientMaker);

} // namespace dragon
