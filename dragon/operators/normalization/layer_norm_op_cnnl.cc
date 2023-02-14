#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/normalization/layer_norm_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLLayerNormOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto &W = Input(1), &B = Input(2);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto N = X.count(0, axis);
  const auto C = X.count(axis);
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({C}), T);
  INITIALIZE_TENSOR_VIA_SPEC(B, vec64_t({C}), T);
  auto* X_mu = Output("X_mu")->Reshape({N});
  auto* X_rsig = Output("X_rsig")->Reshape({N});

  CNNLSetTensorDesc<T>(input_desc_, {N, C});
  CNNLSetTensorDesc<T>(scale_desc_, {C});
  CNNLSetTensorDesc<T>(stats_desc_, {N});

  size_t workspace_size = 0;
  CNNL_CHECK(cnnlGetLayerNormOpWorkspaceSize(
      ctx()->cnnl_handle(), 1, input_desc_, &workspace_size));
  CNNL_CHECK(cnnlLayerNormForward(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      1,
      scale_desc_,
      W.template data<T, Context>(),
      B.template data<T, Context>(),
      epsilon_,
      ctx()->workspace()->template data<Context>(workspace_size),
      workspace_size,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      stats_desc_,
      X_mu->template mutable_data<T, Context>(),
      X_rsig->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLLayerNormGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto N = X.count(0, axis);
  const auto C = X.count(axis);
  auto &X_mu = Input("X_mu"), &X_rsig = Input("X_rsig");

  CNNLSetTensorDesc<T>(this->input_desc_, {N, C});
  CNNLSetTensorDesc<T>(this->scale_desc_, {C});
  CNNLSetTensorDesc<T>(this->stats_desc_, {N});

  size_t workspace_size = 0;
  CNNL_CHECK(cnnlGetLayerNormBackwardWorkspaceSize(
      ctx()->cnnl_handle(), this->input_desc_, 1, &workspace_size));
  CNNL_CHECK(cnnlLayerNormBackward_v2(
      ctx()->cnnl_handle(),
      this->input_desc_,
      X.template data<T, Context>(),
      1,
      this->input_desc_,
      dY.template data<T, Context>(),
      this->scale_desc_,
      W.template data<T, Context>(),
      this->stats_desc_,
      X_mu.template data<T, Context>(),
      X_rsig.template data<T, Context>(),
      ctx()->workspace()->template data<Context>(workspace_size),
      workspace_size,
      this->input_desc_,
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      dW->Reshape({C})->template mutable_data<T, Context>(),
      dB->Reshape({C})->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(LayerNorm);
DEPLOY_CNNL_OPERATOR(LayerNormGradient);

} // namespace dragon

#endif // USE_MLU
