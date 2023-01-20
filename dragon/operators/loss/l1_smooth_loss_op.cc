#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/loss/l1_loss_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SmoothL1LossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);

  const auto N = X.count();
  CHECK_EQ(Y.count(), N) << "\nNumel of X and Y must be matched.";
  auto* loss = ctx()->workspace()->template data<T, Context>(N);

  kernels::SmoothL1Loss(
      N,
      beta_,
      X.template data<T, Context>(),
      Y.template data<T, Context>(),
      loss,
      ctx());

  // Reduction.
  if (reduction_ == "NONE") {
    L->ReshapeLike(X);
    math::Copy(N, loss, L->template mutable_data<T, Context>(), ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_MEAN") {
      normalizer *= X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer *= N;
    }
    kernels::ReduceLoss(
        N,
        0,
        normalizer,
        loss,
        (T*)nullptr,
        L->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void SmoothL1LossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dL = Input(2);
  auto* dX = Output(0)->ReshapeLike(X);

  const auto N = X.count();
  auto* dx = dX->template mutable_data<T, Context>();

  kernels::SmoothL1LossGrad(
      N,
      beta_,
      X.template data<T, Context>(),
      Y.template data<T, Context>(),
      dx,
      ctx());

  // Gradient w.r.t. the first input.
  if (reduction_ == "NONE") {
    math::Mul(N, dL.template data<T, Context>(), dx, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_MEAN") {
      normalizer *= X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer *= N;
    }
    kernels::ReduceLossGrad(
        N,
        0,
        normalizer,
        dL.template data<T, Context>(),
        (T*)nullptr,
        dx,
        ctx());
  }

  // Gradient w.r.t. the second input.
  if (OutputSize() > 1 && Output(1)->has_name()) {
    auto* dY = Output(1)->ReshapeLike(Input(1));
    math::Neg(N, dx, dY->template mutable_data<T, Context>(), ctx());
  }
}

DEPLOY_CPU_OPERATOR(SmoothL1Loss);
DEPLOY_CPU_OPERATOR(SmoothL1LossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SmoothL1Loss);
DEPLOY_CUDA_OPERATOR(SmoothL1LossGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(SmoothL1Loss, SmoothL1Loss);
DEPLOY_MPS_OPERATOR(SmoothL1LossGradient, SmoothL1LossGradient);
#endif

OPERATOR_SCHEMA(SmoothL1Loss)
    /* X, Y */
    .NumInputs(2)
    /* L */
    .NumOutputs(1);

OPERATOR_SCHEMA(SmoothL1LossGradient)
    /* X, Y, dL */
    .NumInputs(3)
    /* dX, dY */
    .NumOutputs(1, 2);

REGISTER_GRADIENT(SmoothL1Loss, GenericGradientMaker);

} // namespace dragon
