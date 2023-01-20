#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/loss/l1_loss_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLSmoothL1LossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);

  const auto N = X.count();
  CHECK_EQ(Y.count(), N) << "\nNumel of X and Y must be matched.";
  auto* loss = ctx()->workspace()->template data<T, Context>(N, "BufferKernel");

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
    reduce_impl_.Setup<T>({N}, {0}, ctx());
    reduce_impl_.Compute<T>(
        loss,
        L->Reshape({})->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(reduce_impl_.scratch_size()),
        ctx(),
        1.f / float(normalizer));
  }
}

template <class Context>
template <typename T>
void CNNLSmoothL1LossGradientOp<Context>::DoRunWithType() {
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
    const auto scale = 1.f / float(normalizer);
    auto* dl = ctx()->workspace()->template data<T, Context>(1);
    math::Scale(1, scale, dL.template data<T, Context>(), dl, ctx());
    math::Mul(1, &N, 1, vec64_t({1}).data(), dx, dl, dx, ctx());
  }

  // Gradient w.r.t. the second input.
  if (OutputSize() > 1 && Output(1)->has_name()) {
    auto* dY = Output(1)->ReshapeLike(Input(1));
    math::Neg(N, dx, dY->template mutable_data<T, Context>(), ctx());
  }
}

DEPLOY_CNNL_OPERATOR(SmoothL1Loss);
DEPLOY_CNNL_OPERATOR(SmoothL1LossGradient);

} // namespace dragon

#endif // USE_MLU
