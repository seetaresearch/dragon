#include "dragon/core/workspace.h"
#include "dragon/operators/loss/l1_loss_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SmoothL1LossOp<Context>::DoRunWithType() {
  auto &X = Input(0), *L = Output(0);

  const auto N = X.count();
  for (int i = 1; i < InputSize(); i++) {
    CHECK_EQ(Input(i).count(), N)
        << "\nTensor(" << Input(i).name() << ") takes the "
        << "dimensions of " << Input(i).DimString() << ", "
        << "while " << X.DimString() << " is required.";
  }

  auto* scratch = ctx()->workspace()->template data<T, Context>(N);

  if (InputSize() > 1) {
    math::Sub(
        N,
        X.template data<T, Context>(),
        Input(1).template data<T, Context>(),
        scratch,
        ctx());
    kernels::SmoothL1(N, beta_, scratch, scratch, ctx());
  } else {
    kernels::SmoothL1(N, beta_, X.template data<T, Context>(), scratch, ctx());
  }

  // Reduction.
  if (reduction_ == "NONE") {
    math::Copy(
        N,
        scratch,
        L->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
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
        scratch,
        (T*)nullptr,
        L->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void SmoothL1LossGradientOp<Context>::DoRunWithType() {
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
    kernels::SmoothL1Grad(N, beta_, dx, dx, ctx());
  } else {
    kernels::SmoothL1Grad(N, beta_, X.template data<T, Context>(), dx, ctx());
  }

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
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SmoothL1Loss);
#endif

DEPLOY_CPU_OPERATOR(SmoothL1LossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SmoothL1LossGradient);
#endif

OPERATOR_SCHEMA(SmoothL1Loss)
    /* X, Y */
    .NumInputs(1, 2)
    /* L */
    .NumOutputs(1);

OPERATOR_SCHEMA(SmoothL1LossGradient)
    /* X, Y, dL */
    .NumInputs(2, 3)
    /* dX, dY */
    .NumOutputs(1, 2);

REGISTER_GRADIENT(SmoothL1Loss, GenericGradientMaker);

} // namespace dragon
