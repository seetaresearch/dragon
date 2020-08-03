#ifdef USE_CUDNN

#include "dragon/operators/activation/tanh_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNTanhOp<Context>::DoRunWithType() {
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
void CuDNNTanhOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNTanhGradientOp<Context>::DoRunWithType() {
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
void CuDNNTanhGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CUDNN(Tanh);
DEPLOY_CUDNN(TanhGradient);

} // namespace dragon

#endif // USE_CUDNN
