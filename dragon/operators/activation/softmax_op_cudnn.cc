#ifdef USE_CUDNN

#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/activation/softmax_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNSoftmaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  const auto C = X.dim(axis);
  const auto N = X.count(0, axis), S = X.count(axis + 1);
  if (C < 384 && S == 1) {
    kernels::Softmax(
        N,
        S,
        C,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
    return;
  }
  CuDNNSetTensorDesc<T>(input_desc_, {N, C, S, 1});
  CUDNN_CHECK(cudnnSoftmaxForward(
      ctx()->cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE,
      S == 1 ? CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL,
      CuDNNTraits<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CuDNNTraits<T>::zero,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CuDNNSoftmaxGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, Y.ndim(), -1);
  const auto C = Y.dim(axis);
  const auto N = Y.count(0, axis), S = Y.count(axis + 1);
  if (C < 256 && S == 1) {
    kernels::SoftmaxGrad(
        Y.count(0, axis),
        Y.count(axis + 1),
        Y.dim(axis),
        dY.template data<T, Context>(),
        Y.template data<T, Context>(),
        dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
        ctx());
    return;
  }
  CuDNNSetTensorDesc<T>(input_desc_, {N, C, S, 1});
  CUDNN_CHECK(cudnnSoftmaxBackward(
      ctx()->cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE,
      S == 1 ? CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL,
      CuDNNTraits<T>::one,
      input_desc_,
      Y.template data<T, Context>(),
      input_desc_,
      dY.template data<T, Context>(),
      CuDNNTraits<T>::zero,
      input_desc_,
      dX->ReshapeLike(Y)->template mutable_data<T, Context>()));
}

DEPLOY_CUDNN_OPERATOR(Softmax);
DEPLOY_CUDNN_OPERATOR(SoftmaxGradient);

} // namespace dragon

#endif // USE_CUDNN
