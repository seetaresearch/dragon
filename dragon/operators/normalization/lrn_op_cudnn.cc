#ifdef USE_CUDNN

#include "dragon/operators/normalization/lrn_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNLRNOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CuDNNSetTensorDesc<T>(&input_desc_, X.dims(), data_format());

  CUDNN_CHECK(cudnnLRNCrossChannelForward(
      ctx()->cudnn_handle(),
      lrn_desc_,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      CuDNNType<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
void CuDNNLRNOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNLRNGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dY = Input(2), *dX = Output(0);
  CuDNNSetTensorDesc<T>(&input_desc_, X.dims(), data_format());

  CUDNN_CHECK(cudnnLRNCrossChannelBackward(
      ctx()->cudnn_handle(),
      lrn_desc_,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      CuDNNType<T>::one,
      input_desc_,
      Y.template data<T, Context>(),
      input_desc_,
      dY.template data<T, Context>(),
      input_desc_,
      X.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      dX->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
void CuDNNLRNGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(LRN);
DEPLOY_CUDNN_OPERATOR(LRNGradient);

} // namespace dragon

#endif // USE_CUDNN
