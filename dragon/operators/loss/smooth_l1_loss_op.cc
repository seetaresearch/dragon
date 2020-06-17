#include "dragon/core/workspace.h"
#include "dragon/operators/loss/l1_loss_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SmoothL1LossOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  for (int i = 1; i < InputSize(); i++) {
    CHECK_EQ(X.count(), Input(i).count())
        << "\nTensor(" << Input(i).name() << ") takes the "
        << "dimensions of " << Input(i).DimString() << ", "
        << "while " << X.DimString() << " is required.";
  }

  // Allocate a temporal error buffer
  auto* x_error = ws()->template data<T, Context>({X.count()})[0];

  // Compute the error of inputs
  if (InputSize() > 1) {
    STORE_INPUT_SPEC(1);
    math::Sub(
        X.count(),
        X.template data<T, Context>(),
        Input(1).template data<T, Context>(),
        x_error,
        ctx());
  } else {
    math::Copy(X.count(), X.template data<T, Context>(), x_error, ctx());
  }

  // Compute the smoothed absolute error
  kernel::SmoothL1(X.count(), beta_, x_error, x_error, ctx());

  // Reduction
  if (reduction_ == "NONE") {
    math::Copy(
        X.count(),
        x_error,
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_SIZE") {
      normalizer *= X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer *= X.count();
    }
    kernel::ReduceLoss(
        X.count(),
        0,
        normalizer,
        x_error,
        nullptr,
        Y->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void SmoothL1LossOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SmoothL1LossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(-1), *dX = Output(0);

  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();

  // Compute the error of inputs
  if (InputSize() > 2) {
    math::Sub(
        dX->count(),
        X.template data<T, Context>(),
        Input(1).template data<T, Context>(),
        dx,
        ctx());
  } else {
    math::Copy(dX->count(), X.template data<T, Context>(), dx, ctx());
  }

  // Compute the partial gradient
  kernel::SmoothL1Grad(dX->count(), beta_, dx, dx, ctx());

  // Gradient w.r.t. the first input
  if (reduction_ == "NONE") {
    math::Mul(dX->count(), dy, dx, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_SIZE") {
      normalizer *= dX->dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer *= dX->count();
    }
    kernel::ReduceLossGrad(dX->count(), 0, normalizer, dy, nullptr, dx, ctx());
  }

  // Gradient w.r.t. the second input
  if (OutputSize() > 1 && Output(1)->has_name()) {
    Output(1)->ReshapeLike(Input(1));
    math::Scale(
        dX->count(),
        -1.f,
        dx,
        Output(1)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void SmoothL1LossGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(SmoothL1Loss);
#ifdef USE_CUDA
DEPLOY_CUDA(SmoothL1Loss);
#endif

DEPLOY_CPU(SmoothL1LossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(SmoothL1LossGradient);
#endif

OPERATOR_SCHEMA(SmoothL1Loss)
    /* X, T */
    .NumInputs(1, 2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SmoothL1LossGradient)
    /* X, T, dY */
    .NumInputs(2, 3)
    /* dX, dT */
    .NumOutputs(1, 2);

REGISTER_GRADIENT(SmoothL1Loss, GenericGradientMaker);

} // namespace dragon
