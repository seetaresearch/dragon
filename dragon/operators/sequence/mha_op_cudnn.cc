#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/sequence/mha_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNMultiHeadSelfAttnOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  if (this->in_channels_ != X.dim(1)) this->template SetOpDesc<T>();
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({this->weight_count_}), T);

  auto* seqlens = Output("X_seqlens")
                      ->Reshape({X.dim(0)})
                      ->template mutable_data<int, Context>();
  math::Set(X.dim(0), int(X.dim(1)), seqlens, ctx());

  auto workspace_size = phase() == "TRAIN" ? this->train_workspace_size_
                                           : this->infer_workspace_size_;

  CUDNN_CHECK(cudnnMultiHeadAttnForward(
      ctx()->cudnn_handle(),
      this->attn_desc_,
      -1, // currIdx
      vec32_t(X.dim(1), 0).data(), // loWinIdx
      vec32_t(X.dim(1), X.dim(1)).data(), // hiWinIdx
      seqlens, // QO_seqlens
      seqlens, // KV_seqlens
      this->input_desc_,
      X.template data<T, Context>(), // Q
      nullptr, // residuals
      this->input_desc_,
      X.template data<T, Context>(), // K
      this->input_desc_,
      X.template data<T, Context>(), // V
      this->output_desc_,
      Y->Reshape(this->output_dims_)->template mutable_data<T, Context>(),
      sizeof(T) * this->weight_count_,
      W.template data<T, Context>(),
      workspace_size,
      ctx()->workspace()->template data<Context>(workspace_size),
      phase() == "TRAIN" ? this->reserve_size_ : 0,
      phase() == "TRAIN" ? Output("X_reserve")
                               ->Reshape({int64_t(this->reserve_size_)})
                               ->template mutable_data<uint8_t, Context>()
                         : nullptr));
}

template <class Context>
template <typename T>
void CuDNNMultiHeadSelfAttnGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  auto &X_reserve = Input("X_reserve"), *dX = Output(0), *dW = Output(1);
  if (this->in_channels_ != X.dim(1)) this->template SetOpDesc<T>();

  T* dk_and_dv = nullptr;
  auto* seqlens = Input("X_seqlens").template data<int, Context>();

  if (dX->has_name() || dW->has_name()) {
    dk_and_dv = ctx()->workspace()->template data<T, Context>({2 * X.count()});
    CUDNN_CHECK(cudnnMultiHeadAttnBackwardData(
        ctx()->cudnn_handle(),
        this->attn_desc_,
        vec32_t(X.dim(1), 0).data(), // loWinIdx
        vec32_t(X.dim(1), X.dim(1)).data(), // hiWinIdx
        seqlens, // QO_seqlens
        seqlens, // KV_seqlens
        this->output_desc_,
        dY.template data<T, Context>(),
        this->input_desc_,
        dX->ReshapeLike(X)->template mutable_data<T, Context>(), // dQ
        X.template data<T, Context>(), // Q
        this->input_desc_,
        dk_and_dv, // dK
        X.template data<T, Context>(), // K
        this->input_desc_,
        dk_and_dv + X.count(), // dV
        X.template data<T, Context>(), // V
        sizeof(T) * this->weight_count_,
        W.template data<T, Context>(),
        this->train_workspace_size_,
        ctx()->workspace()->template data<Context>(this->train_workspace_size_),
        this->reserve_size_,
        X_reserve.template mutable_data<uint8_t, Context>()));
  }

  if (dW->has_name()) {
    CUDNN_CHECK(cudnnMultiHeadAttnBackwardWeights(
        ctx()->cudnn_handle(),
        this->attn_desc_,
        CUDNN_WGRAD_MODE_SET,
        this->input_desc_,
        X.template data<T, Context>(), // Q
        this->input_desc_,
        X.template data<T, Context>(), // K
        this->input_desc_,
        X.template data<T, Context>(), // V
        this->output_desc_,
        dY.template data<T, Context>(),
        sizeof(T) * this->weight_count_,
        W.template data<T, Context>(),
        dW->ReshapeLike(W)->template mutable_data<T, Context>(),
        this->train_workspace_size_,
        ctx()->workspace()->template data<Context>(this->train_workspace_size_),
        this->reserve_size_,
        X_reserve.template mutable_data<uint8_t, Context>()));
  }

  for (int i = 0; i < 2 && dX->has_name(); ++i) {
    math::Add(
        X.count(),
        dX->template data<T, Context>(),
        dk_and_dv + i * X.count(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CUDNN_OPERATOR(MultiHeadSelfAttn);
DEPLOY_CUDNN_OPERATOR(MultiHeadSelfAttnGradient);

} // namespace dragon

#endif // USE_CUDNN
