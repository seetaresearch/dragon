#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/normalization/group_norm_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLGroupNormOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto &W = Input(1), &B = Input(2);
  GetBaseArguments();
  CHECK_EQ(data_format(), "NHWC");
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({C_}), T);
  INITIALIZE_TENSOR_VIA_SPEC(B, vec64_t({C_}), T);

  CNNLSetTensorDesc<T>(input_desc_, X.dims(), data_format());
  CNNLSetTensorDesc<T>(scale_desc_, {C_});

  size_t workspace_size = 0;
  CNNL_CHECK(cnnlGetGroupNormForwardWorkspaceSize(
      ctx()->cnnl_handle(), G_, input_desc_, &workspace_size));
  CNNL_CHECK(cnnlGroupNormForward_v2(
      ctx()->cnnl_handle(),
      epsilon_,
      G_,
      input_desc_,
      X.template data<T, Context>(),
      scale_desc_,
      W.template data<T, Context>(),
      B.template data<T, Context>(),
      ctx()->workspace()->template data<Context>(workspace_size),
      workspace_size,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLGroupNormGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  GetBaseArguments();
  CHECK_EQ(data_format(), "NHWC");

  auto* X_mu = Output("X_mu")->Reshape({N_, G_});
  auto* X_rsig = Output("X_rsig")->Reshape({N_, G_});
  CNNLSetTensorDesc<T>(this->input_desc_, X.dims(), data_format());
  CNNLSetTensorDesc<T>(this->scale_desc_, {C_});
  CNNLSetTensorDesc<T>(this->stats_desc_, {N_, G_});
  mean_impl_.Setup<T>(CNNL_REDUCE_AVG, {N_, S_, G_, D_}, {1, 3}, ctx());
  rsig_impl_.Setup<T>(CNNL_REDUCE_SUMSQ, {N_, S_, G_, D_}, {1, 3}, ctx());

  auto* x = X.template data<T, Context>();
  auto* mu = X_mu->template mutable_data<T, Context>();
  auto* rsig = X_rsig->template mutable_data<T, Context>();
  auto* dx = dX->ReshapeLike(X)->template mutable_data<T, Context>();
  auto* scratch = ctx()->workspace()->template data<Context>(
      std::max(mean_impl_.scratch_size(), rsig_impl_.scratch_size()));

  // Compute moments.
  const auto NxG = N_ * G_, SxD = S_ * D_;
  mean_impl_.Compute<T>(x, mu, scratch, ctx());
  rsig_impl_.Compute<T>(x, rsig, scratch, ctx());
  math::Scale(NxG, 1.f / float(SxD), rsig, rsig, ctx());

  // Compute rsig.
  math::Square(NxG, mu, dx, ctx());
  math::Sub(NxG, rsig, dx, rsig, ctx());
  math::InvStd(NxG, epsilon_, rsig, rsig, ctx());

  // Gradient w.r.t. gamma, beta and input.
  size_t workspace_size = 0;
  CNNL_CHECK(cnnlGetGroupNormBackwardWorkspaceSize(
      ctx()->cnnl_handle(), N_ * C_, &workspace_size));
  CNNL_CHECK(cnnlGroupNormBackward(
      ctx()->cnnl_handle(),
      this->input_desc_,
      X.template data<T, Context>(),
      this->input_desc_,
      dY.template data<T, Context>(),
      this->scale_desc_,
      W.template data<T, Context>(),
      this->stats_desc_,
      mu,
      this->stats_desc_,
      rsig,
      G_,
      this->input_desc_,
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      this->scale_desc_,
      dW->Reshape({C_})->template mutable_data<T, Context>(),
      this->scale_desc_,
      dB->Reshape({C_})->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(workspace_size),
      workspace_size));
}

DEPLOY_CNNL_OPERATOR(GroupNorm);
DEPLOY_CNNL_OPERATOR(GroupNormGradient);

} // namespace dragon

#endif // USE_MLU
