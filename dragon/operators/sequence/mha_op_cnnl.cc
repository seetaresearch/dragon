#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/sequence/mha_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLMultiHeadSelfAttnOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  if (this->in_channels_ != X.dim(1)) this->template SetOpDesc<T>();
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({this->weight_count_}), T);

  auto workspace_size = phase() == "TRAIN" ? this->train_workspace_size_
                                           : this->infer_workspace_size_;

  CNNL_CHECK(cnnlMultiHeadAttnForward(
      ctx()->cnnl_handle(),
      this->attn_desc_,
      -1,
      nullptr, // padding_mask_desc
      nullptr, // padding_mask
      InputSize() > 2 ? this->attn_mask_desc_ : nullptr,
      InputSize() > 2 ? Input(2).template data<T, Context>() : nullptr,
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
void CNNLMultiHeadSelfAttnGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  auto &X_reserve = Input("X_reserve"), *dX = Output(0), *dW = Output(1);
  if (this->in_channels_ != X.dim(1)) this->template SetOpDesc<T>();

  T* dk_and_dv = nullptr;

  if (dX->has_name() || dW->has_name()) {
    dk_and_dv = ctx()->workspace()->template data<T, Context>({2 * X.count()});
    CNNL_CHECK(cnnlMultiHeadAttnBackwardData(
        ctx()->cnnl_handle(),
        this->attn_desc_,
        nullptr, // padding_mask_desc
        nullptr, // padding_mask
        Input(2).has_name() ? this->attn_mask_desc_ : nullptr,
        Input(2).has_name() ? Input(2).template data<T, Context>() : nullptr,
        this->output_desc_,
        dY.template data<T, Context>(),
        this->input_desc_,
        dX->ReshapeLike(X)->template mutable_data<T, Context>(), // dQ
        this->input_desc_,
        dk_and_dv, // dK
        this->input_desc_,
        dk_and_dv + X.count(), // dV
        sizeof(T) * this->weight_count_,
        W.template data<T, Context>(),
        this->train_workspace_size_,
        ctx()->workspace()->template data<Context>(this->train_workspace_size_),
        this->reserve_size_,
        X_reserve.template mutable_data<uint8_t, Context>()));
  }

  if (dW->has_name()) {
    CNNL_CHECK(cnnlMultiHeadAttnBackwardWeights(
        ctx()->cnnl_handle(),
        this->attn_desc_,
        CNNL_WGRAD_MODE_SET,
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

DEPLOY_CNNL_OPERATOR(MultiHeadSelfAttn);
DEPLOY_CNNL_OPERATOR(MultiHeadSelfAttnGradient);

} // namespace dragon

#endif // USE_MLU
