#ifdef USE_CUDNN

#include "dragon/operators/activation/softmax_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNSoftmaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  CuDNNSetTensorDesc<T>(
      &input_desc_, {X.count(0, axis), X.dim(axis), X.count(axis + 1)});

  CUDNN_CHECK(cudnnSoftmaxForward(
      ctx()->cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL,
      CuDNNType<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CuDNNSoftmaxGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(Y);

  CuDNNSetTensorDesc<T>(
      &input_desc_, {Y.count(0, axis), Y.dim(axis), Y.count(axis + 1)});

  CUDNN_CHECK(cudnnSoftmaxBackward(
      ctx()->cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL,
      CuDNNType<T>::one,
      input_desc_,
      Y.template data<T, Context>(),
      input_desc_,
      dY.template data<T, Context>(),
      CuDNNType<T>::zero,
      input_desc_,
      dX->ReshapeLike(Y)->template mutable_data<T, Context>()));
}

DEPLOY_CUDNN(Softmax);
DEPLOY_CUDNN(SoftmaxGradient);

} // namespace dragon

#endif // USE_CUDNN
