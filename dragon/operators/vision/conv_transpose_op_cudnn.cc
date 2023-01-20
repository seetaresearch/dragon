#ifdef USE_CUDNN

#include "dragon/operators/vision/conv_transpose_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNConvTransposeOp<Context>::DoRunWithType() {
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
      Y->dims(),
      W.dims(),
      X.dims(),
      ctx());

  Y_impl_.Compute<T>(
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());

  if (HasBias()) {
    CuDNNSetBiasDesc<T>(Y_impl_.B_desc_, Y->ndim(), W.dim(1), data_format());
    CUDNN_CHECK(cudnnAddTensor(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        Y_impl_.B_desc_,
        Input(2).template data<T, Context>(),
        CuDNNType<T>::one,
        Y_impl_.X_desc_,
        Y->template mutable_data<T, Context>()));
  }
}

template <class Context>
template <typename T>
void CuDNNConvTransposeGradientOp<Context>::DoRunWithType() {
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
        dY.dims(),
        W.dims(),
        X.dims(),
        ctx());
  }

  if (dW->has_name()) {
    dW_impl_.Setup<T>(
        pads_begin_,
        strides_,
        dilations_,
        group_,
        data_format(),
        dY.dims(),
        W.dims(),
        X.dims(),
        ctx());
  }

  if (dX->has_name()) {
    dX_impl_.Compute<T>(
        dY.template data<T, Context>(),
        W.template data<T, Context>(),
        nullptr,
        dX->template mutable_data<T, Context>(),
        ctx());
  }

  if (dW->has_name() || HasBias()) {
    dW_impl_.Compute<T>(
        X.template data<T, Context>(),
        dY.template data<T, Context>(),
        dW->template mutable_data<T, Context>(),
        nullptr,
        ctx());
  }

  if (HasBias()) {
    CuDNNSetBiasDesc<T>(dW_impl_.B_desc_, dY.ndim(), W.dim(1), data_format());
    CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        dW_impl_.X_desc_,
        dY.template data<T, Context>(),
        CuDNNType<T>::zero,
        dW_impl_.B_desc_,
        Output(2)->template mutable_data<T, Context>()));
  }
}

DEPLOY_CUDNN_OPERATOR(ConvTranspose);
DEPLOY_CUDNN_OPERATOR(ConvTransposeGradient);

} // namespace dragon

#endif // USE_CUDNN
