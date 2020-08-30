#include "dragon/core/workspace.h"
#include "dragon/operators/loss/sigmoid_loss_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SigmoidCrossEntropyOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  CHECK_EQ(X.count(), Input(1).count())
      << "\nNumber of preds must match the number of targets.";

  auto scratches = ws()->template data<Context>({
      X.count() * sizeof(T), // loss
      X.count() * sizeof(int), // mask
  });
  auto* loss = static_cast<T*>(scratches[0]);
  auto* mask = static_cast<int*>(scratches[1]);

  kernel::SigmoidCrossEntropy(
      X.count(),
      X.template data<T, Context>(),
      Input(1).template data<T, Context>(),
      loss,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    math::Copy(
        X.count(),
        loss,
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
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
        Y->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void SigmoidCrossEntropyOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SigmoidCrossEntropyGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(-1), *dX = Output(0);

  auto* mask = ws()->template data<int, Context>({dX->count()})[0];
  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();

  kernel::SigmoidCrossEntropyGrad(
      dX->count(),
      X.template data<T, Context>(),
      Input(1).template data<T, Context>(),
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
void SigmoidCrossEntropyGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(SigmoidCrossEntropy);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SigmoidCrossEntropy);
#endif

DEPLOY_CPU_OPERATOR(SigmoidCrossEntropyGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SigmoidCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SigmoidCrossEntropy)
    /* X, T */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SigmoidCrossEntropyGradient)
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

REGISTER_GRADIENT(SigmoidCrossEntropy, GradientMaker);

} // namespace dragon
