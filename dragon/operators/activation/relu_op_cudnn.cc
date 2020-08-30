#ifdef USE_CUDNN

#include "dragon/operators/activation/relu_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNReluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  CuDNNSetTensorDesc<T>(&input_desc_, X.dims());
  CUDNN_CHECK(cudnnActivationForward(
      ctx()->cudnn_handle(),
      act_desc_,
      CuDNNType<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
void CuDNNReluOp<Context>::RunOnDevice() {
  if (this->alpha_ != 0.f) {
    // CuDNN does not support LeakyReLU
    return ReluOp<Context>::RunOnDevice();
  }
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNReluGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  CuDNNSetTensorDesc<T>(&input_desc_, Y.dims());
  CUDNN_CHECK(cudnnActivationBackward(
      ctx()->cudnn_handle(),
      act_desc_,
      CuDNNType<T>::one,
      input_desc_,
      Y.template data<T, Context>(),
      input_desc_,
      dY.template data<T, Context>(),
      input_desc_,
      Y.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      dX->ReshapeLike(Y)->template mutable_data<T, Context>()));
}

template <class Context>
void CuDNNReluGradientOp<Context>::RunOnDevice() {
  if (this->alpha_ != 0.f || this->max_value_ > 0.f) {
    // CuDNN does not support LeakyReLU and ClippedReLU
    return ReluGradientOp<Context>::RunOnDevice();
  }
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(Relu);
DEPLOY_CUDNN_OPERATOR(ReluGradient);

} // namespace dragon

#endif // USE_CUDNN
