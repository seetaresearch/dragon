#ifdef USE_MLU

#include "dragon/operators/activation/log_softmax_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLLogSoftmaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  const auto C = X.dim(axis);
  const auto N = X.count(0, axis), S = X.count(axis + 1);
  const auto X_dims = (S == 1 ? vec64_t({N, 1, C}) : vec64_t({N, C, S}));
  CNNLSetTensorDesc<T>(input_desc_, X_dims);
  CNNL_CHECK(cnnlSoftmaxForward_v2(
      ctx()->cnnl_handle(),
      CNNL_SOFTMAX_LOG,
      S == 1 ? CNNL_SOFTMAX_MODE_LOW_DIMENSION
             : CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION,
      CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
      nullptr, // alpha
      input_desc_,
      X.template data<T, Context>(),
      nullptr, // beta
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLLogSoftmaxGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, Y.ndim(), -1);
  const auto C = Y.dim(axis);
  const auto N = Y.count(0, axis), S = Y.count(axis + 1);
  const auto Y_dims = (S == 1 ? vec64_t({N, 1, C}) : vec64_t({N, C, S}));
  CNNLSetTensorDesc<T>(this->input_desc_, Y_dims);
  CNNL_CHECK(cnnlSoftmaxBackward(
      ctx()->cnnl_handle(),
      CNNL_SOFTMAX_LOG,
      S == 1 ? CNNL_SOFTMAX_MODE_LOW_DIMENSION
             : CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION,
      nullptr, // alpha
      this->input_desc_,
      Y.template data<T, Context>(),
      this->input_desc_,
      dY.template data<T, Context>(),
      nullptr, // beta
      this->input_desc_,
      dX->ReshapeLike(Y)->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(LogSoftmax);
DEPLOY_CNNL_OPERATOR(LogSoftmaxGradient);

} // namespace dragon

#endif // USE_MLU
