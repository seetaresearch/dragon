#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/loss/cross_entropy_loss_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLSigmoidCrossEntropyLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);

  const auto N = X.count();
  CHECK_EQ(Y.count(), N) << "\nNumel of X and Y must be matched.";

  auto* scratch = ctx()->workspace()->template data<T, Context>(N * 2 + 1);
  auto *loss = scratch, *mask = scratch + N;

  kernels::SigmoidCrossEntropy(
      N,
      X.template data<T, Context>(),
      Y.template data<T, Context>(),
      loss,
      mask,
      ctx());

  if (reduction_ == "NONE") {
    L->ReshapeLike(X);
    math::Copy(N, loss, L->template mutable_data<T, Context>(), ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = N;
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
void CNNLSigmoidCrossEntropyLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dL = Input(2);
  auto* dX = Output(0)->ReshapeLike(X);

  const auto N = X.count();
  auto* dl = dL.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  auto* mask = ctx()->workspace()->template data<T, Context>(N + 1);

  kernels::SigmoidCrossEntropyGrad(
      N,
      X.template data<T, Context>(),
      Y.template data<T, Context>(),
      dx,
      mask,
      ctx());

  if (this->reduction_ == "NONE") {
    math::Mul(N, dl, dx, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (this->reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (this->reduction_ == "MEAN") {
      normalizer = N;
    }
    const auto scale = 1.f / float(normalizer);
    auto* dl = ctx()->workspace()->template data<T, Context>(1);
    math::Scale(1, scale, dL.template data<T, Context>(), dl, ctx());
    math::Mul(1, &N, 1, vec64_t({1}).data(), dx, dl, dx, ctx());
  }
}

DEPLOY_CNNL_OPERATOR(SigmoidCrossEntropyLoss);
DEPLOY_CNNL_OPERATOR(SigmoidCrossEntropyLossGradient);

} // namespace dragon

#endif // USE_MLU
