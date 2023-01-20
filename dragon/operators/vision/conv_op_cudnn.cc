#ifdef USE_CUDNN

#include "dragon/operators/vision/conv_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNConvOp<Context>::DoRunWithType() {
  ConvOpBase<Context>::Reshape();
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  INITIALIZE_TENSOR_VIA_SPEC(W, w_shape_, T);
  if (HasBias()) INITIALIZE_TENSOR_VIA_SPEC(Input(2), b_shape_, T);

  Y_impl_.Setup<T>(
      pads_begin_,
      strides_,
      dilations_,
      group_,
      data_format(),
      X.dims(),
      W.dims(),
      Y->dims(),
      ctx());

  Y_impl_.Compute<T>(
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      HasBias() ? Input(2).template data<T, Context>() : nullptr,
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void CuDNNConvGradientOp<Context>::DoRunWithType() {
  ConvOpBase<Context>::Reshape(true);
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  auto *dX = Output(0), *dW = Output(1);

  if (dX->has_name()) {
    dX_impl_.Setup<T>(
        pads_begin_,
        strides_,
        dilations_,
        group_,
        data_format(),
        X.dims(),
        W.dims(),
        dY.dims(),
        ctx());
  }

  if (dW->has_name()) {
    dW_impl_.Setup<T>(
        pads_begin_,
        strides_,
        dilations_,
        group_,
        data_format(),
        X.dims(),
        W.dims(),
        dY.dims(),
        ctx());
  }

  if (dX->has_name()) {
    dX_impl_.Compute<T>(
        dY.template data<T, Context>(),
        W.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }

  if (dW->has_name()) {
    dW_impl_.Compute<T>(
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dW->template mutable_data<T, Context>(),
        HasBias() ? Output(2)->template mutable_data<T, Context>() : nullptr,
        ctx());
  }
}

DEPLOY_CUDNN_OPERATOR(Conv);
DEPLOY_CUDNN_OPERATOR(ConvGradient);

} // namespace dragon

#endif // USE_CUDNN
