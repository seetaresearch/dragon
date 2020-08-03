#include "dragon/core/workspace.h"
#include "dragon/operators/normalization/batch_norm_op.h"
#include "dragon/utils/filler.h"

#ifdef USE_CUDNN

namespace dragon {

template <class Context>
template <typename T>
void CuDNNBatchNormOp<Context>::DoRunWithType() {
  typedef typename CuDNNType<T>::BNParamType ParamType;

  // Determine the bn desc
  if (Input(0).ndim() == 2) {
    bn_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    CuDNNSetTensorDesc<T>(&input_desc_, vec64_t({N_, C_, 1, 1}));
  } else {
    bn_mode_ = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION_MIN(7, 0, 0)
    if (is_training_ > 0) {
      bn_mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    }
#endif
    CuDNNSetTensorDesc<T>(&input_desc_, Input(0).dims(), data_format());
  }

  // Derive the bn desc
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc_, input_desc_, bn_mode_));

  TENSOR_FILL_WITH_TYPE(Input(1), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(2), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(3), vec64_t({C_}), ParamType);
  TENSOR_FILL_WITH_TYPE(Input(4), vec64_t({C_}), ParamType);

  if (is_training_ > 0) {
    auto* X_mu = Buffer("X_mu")->Reshape({C_});
    auto* X_rsig = Buffer("X_rsig")->Reshape({C_});
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
        Input(1).template data<ParamType, Context>(), // gamma
        Input(2).template data<ParamType, Context>(), // beta
        is_recomputing_ ? 0.f : 1.f - this->momentum_,
        Input(3).template mutable_data<ParamType, Context>(), // rm
        Input(4).template mutable_data<ParamType, Context>(), // rv
        eps64_,
        X_mu->template mutable_data<ParamType, Context>(), // sm
        X_rsig->template mutable_data<ParamType, Context>())); // sv
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
        Input(1).template data<ParamType, Context>(), // gamma
        Input(2).template data<ParamType, Context>(), // beta
        Input(3).template data<ParamType, Context>(), // rm
        Input(4).template data<ParamType, Context>(), // rv
        eps64_));
  }
}

template <class Context>
void CuDNNBatchNormOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Get the recomputing flag
  auto* flag = ws()->GetTensor("/share/flag/recomputing");
  is_recomputing_ = flag->template data<bool, CPUContext>()[0];

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (XIsType(Input(0), float)) {
    DoRunWithType<float>();
  } else if (XIsType(Input(0), float16)) {
    DoRunWithType<float16>();
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float16", "float32"});
  }
}

template <class Context>
template <typename T>
void CuDNNBatchNormGradientOp<Context>::TrainingImpl() {
  typedef typename CuDNNType<T>::BNParamType ParamType;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto *X_mu = Buffer("X_mu"), *X_rsig = Buffer("X_rsig");

  // Determine the bn desc
  if (Input(0).ndim() == 2) {
    bn_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    CuDNNSetTensorDesc<T>(&input_desc_, vec64_t({N_, C_, 1, 1}));
  } else {
    bn_mode_ = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION_MIN(7, 0, 0)
    bn_mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
    CuDNNSetTensorDesc<T>(&input_desc_, Input(0).dims(), data_format());
  }

  // Derive the bn desc
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc_, input_desc_, bn_mode_));

  // Gradient w.r.t. gamma, beta and input
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
      Input(1).template data<ParamType, Context>(), // gamma
      dW->Reshape({C_})->template mutable_data<ParamType, Context>(), // dw
      dB->Reshape({C_})->template mutable_data<ParamType, Context>(), // db
      eps64_,
      X_mu->template data<ParamType, Context>(), // mu
      X_rsig->template data<ParamType, Context>())); // rsig
}

template <class Context>
void CuDNNBatchNormGradientOp<Context>::RunOnDevice() {
  DetermineBaseArguments();

  // Dispatch the training or inference impl
  Output(0)->ReshapeLike(Input(0));
  if (XIsType(Input(0), float)) {
    if (is_training_ > 0) {
      TrainingImpl<float>();
    } else {
      this->template InferenceImpl<float, float>();
    }
  } else if (XIsType(Input(0), float16)) {
    if (is_training_ > 0) {
      TrainingImpl<float16>();
    } else {
      // We will support it some day -:)
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(0).meta()), {"float32"});
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float16", "float32"});
  }
}

DEPLOY_CUDNN(BatchNorm);
DEPLOY_CUDNN(BatchNormGradient);

} // namespace dragon

#endif // USE_CUDNN
