#include "dragon/operators/loss/l2_loss_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void L2LossOp<Context>::DoRunWithType() {
  auto &X = Input(0), *L = Output(0);

  const auto N = X.count();
  auto* loss = ctx()->workspace()->template data<T, Context>(N);

  if (InputSize() > 1) {
    CHECK_EQ(Input(1).count(), N) << "\nNumel of X and Y must be matched.";
    math::Sub(
        N,
        X.template data<T, Context>(),
        Input(1).template data<T, Context>(),
        loss,
        ctx());
    math::Square(N, loss, loss, ctx());
  } else {
    math::Square(N, X.template data<T, Context>(), loss, ctx());
  }

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
void L2LossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dL = Input(-1);
  auto* dX = Output(0)->ReshapeLike(X);

  const auto N = X.count();
  auto* dx = dX->template mutable_data<T, Context>();

  if (InputSize() > 2) {
    math::Sub(
        N,
        X.template data<T, Context>(),
        Input(1).template data<T, Context>(),
        dx,
        ctx());
  } else {
    math::Copy(N, X.template data<T, Context>(), dx, ctx());
  }

  // Gradient w.r.t. the first input.
  if (reduction_ == "NONE") {
    math::Mul(N, dL.template data<T, Context>(), dx, dx, ctx());
    math::Scale(N, 2.f, dx, dx, ctx());
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
        float(normalizer) * 0.5f,
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

DEPLOY_CPU_OPERATOR(L2Loss);
DEPLOY_CPU_OPERATOR(L2LossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(L2Loss);
DEPLOY_CUDA_OPERATOR(L2LossGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(L2Loss, L2Loss);
DEPLOY_MPS_OPERATOR(L2LossGradient, L2LossGradient);
#endif

OPERATOR_SCHEMA(L2Loss)
    /* X, Y */
    .NumInputs(1, 2)
    /* L */
    .NumOutputs(1);

OPERATOR_SCHEMA(L2LossGradient)
    /* X, Y, dL */
    .NumInputs(2, 3)
    /* dX, dY */
    .NumOutputs(1, 2);

REGISTER_GRADIENT(L2Loss, GenericGradientMaker);

} // namespace dragon
