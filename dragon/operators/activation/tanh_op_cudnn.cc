#ifdef USE_CUDNN

#include "dragon/operators/activation/tanh_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNTanhOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  CuDNNSetTensorDesc<T>(input_desc_, X.dims());
  CUDNN_CHECK(cudnnActivationForward(
      ctx()->cudnn_handle(),
      act_desc_,
      CuDNNTraits<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CuDNNTraits<T>::zero,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CuDNNTanhGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  CuDNNSetTensorDesc<T>(this->input_desc_, Y.dims());
  CUDNN_CHECK(cudnnActivationBackward(
      ctx()->cudnn_handle(),
      this->act_desc_,
      CuDNNTraits<T>::one,
      this->input_desc_,
      Y.template data<T, Context>(),
      this->input_desc_,
      dY.template data<T, Context>(),
      this->input_desc_,
      Y.template data<T, Context>(),
      CuDNNTraits<T>::zero,
      this->input_desc_,
      dX->ReshapeLike(Y)->template mutable_data<T, Context>()));
}

DEPLOY_CUDNN_OPERATOR(Tanh);
DEPLOY_CUDNN_OPERATOR(TanhGradient);

} // namespace dragon

#endif // USE_CUDNN
