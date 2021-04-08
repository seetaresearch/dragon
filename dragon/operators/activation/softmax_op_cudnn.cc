#ifdef USE_CUDNN

#include "dragon/operators/activation/softmax_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNSoftmaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  auto S = X.count(axis + 1);
  CuDNNSetTensorDesc<T>(&input_desc_, {X.count(0, axis), X.dim(axis), S, 1});
  CUDNN_CHECK(cudnnSoftmaxForward(
      ctx()->cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE,
      S == 1 ? CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL,
      CuDNNType<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
void CuDNNSoftmaxOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNSoftmaxGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, Y.ndim(), -1);
  auto S = Y.count(axis + 1);
  CuDNNSetTensorDesc<T>(&input_desc_, {Y.count(0, axis), Y.dim(axis), S, 1});
  CUDNN_CHECK(cudnnSoftmaxBackward(
      ctx()->cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE,
      S == 1 ? CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL,
      CuDNNType<T>::one,
      input_desc_,
      Y.template data<T, Context>(),
      input_desc_,
      dY.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      dX->ReshapeLike(Y)->template mutable_data<T, Context>()));
}

template <class Context>
void CuDNNSoftmaxGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(Softmax);
DEPLOY_CUDNN_OPERATOR(SoftmaxGradient);

} // namespace dragon

#endif // USE_CUDNN
