#include "dragon/core/workspace.h"
#include "dragon/operators/normalization/batch_norm_op.h"

#ifdef USE_CUDNN

namespace dragon {

template <class Context>
template <typename T>
void CuDNNBatchNormOp<Context>::DoRunWithType() {
  using ParamT = typename CuDNNType<T>::BNParamType;
  INITIALIZE_TENSOR_VIA_SPEC(Input(1), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(2), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(3), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(4), vec64_t({C_}), ParamT);

  // Set descriptors.
  if (Input(0).ndim() == 2) {
    bn_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    CuDNNSetTensorDesc<T>(&input_desc_, vec64_t({N_, C_, 1, 1}));
  } else {
    bn_mode_ = CUDNN_BATCHNORM_SPATIAL;
    CuDNNSetTensorDesc<T>(&input_desc_, Input(0).dims(), data_format());
  }
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc_, input_desc_, bn_mode_));

  // Run training or inference.
  if (training_ > 0) {
    auto* X_mu = Output("X_mu")->Reshape({C_});
    auto* X_rsig = Output("X_rsig")->Reshape({C_});
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        ctx()->cudnn_handle(),
        bn_mode_,
        CuDNNType<T>::one,
        CuDNNType<T>::zero,
        input_desc_,
        Input(0).template data<T, Context>(), // x
        input_desc_,
        Output(0)->template mutable_data<T, Context>(), // y
        bn_desc_,
        Input(1).template data<ParamT, Context>(), // gamma
        Input(2).template data<ParamT, Context>(), // beta
        1.f - momentum(),
        Input(3).template mutable_data<ParamT, Context>(), // rm
        Input(4).template mutable_data<ParamT, Context>(), // rv
        epsilon_,
        X_mu->template mutable_data<ParamT, Context>(), // sm
        X_rsig->template mutable_data<ParamT, Context>())); // sv
  } else {
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        ctx()->cudnn_handle(),
        bn_mode_,
        CuDNNType<T>::one,
        CuDNNType<T>::zero,
        input_desc_,
        Input(0).template data<T, Context>(),
        input_desc_,
        Output(0)->template mutable_data<T, Context>(), // y
        bn_desc_,
        Input(1).template data<ParamT, Context>(), // gamma
        Input(2).template data<ParamT, Context>(), // beta
        Input(3).template data<ParamT, Context>(), // rm
        Input(4).template data<ParamT, Context>(), // rv
        epsilon_));
  }
}

template <class Context>
template <typename T>
void CuDNNBatchNormGradientOp<Context>::RunTraining() {
  using ParamT = typename CuDNNType<T>::BNParamType;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto &X_mu = Input("X_mu"), &X_rsig = Input("X_rsig");

  // Set descriptors.
  if (Input(0).ndim() == 2) {
    bn_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    CuDNNSetTensorDesc<T>(&input_desc_, vec64_t({N_, C_, 1, 1}));
  } else {
    bn_mode_ = CUDNN_BATCHNORM_SPATIAL;
    CuDNNSetTensorDesc<T>(&input_desc_, Input(0).dims(), data_format());
  }
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc_, input_desc_, bn_mode_));

  // Gradient w.r.t. gamma, beta and input.
  CUDNN_CHECK(cudnnBatchNormalizationBackward(
      ctx()->cudnn_handle(),
      bn_mode_,
      CuDNNType<T>::one,
      CuDNNType<T>::zero,
      CuDNNType<T>::one,
      CuDNNType<T>::zero,
      input_desc_,
      Input(0).template data<T, Context>(), // x
      input_desc_,
      Input(4).template data<T, Context>(), // dy
      input_desc_,
      Output(0)->template mutable_data<T, Context>(), // dx
      bn_desc_,
      Input(1).template data<ParamT, Context>(), // gamma
      dW->Reshape({C_})->template mutable_data<ParamT, Context>(), // dw
      dB->Reshape({C_})->template mutable_data<ParamT, Context>(), // db
      epsilon_,
      X_mu.template data<ParamT, Context>(), // mu
      X_rsig.template data<ParamT, Context>())); // rsig
}

DEPLOY_CUDNN_OPERATOR(BatchNorm);
DEPLOY_CUDNN_OPERATOR(BatchNormGradient);

} // namespace dragon

#endif // USE_CUDNN
