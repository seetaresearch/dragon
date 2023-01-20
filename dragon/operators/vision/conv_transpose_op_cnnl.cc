#ifdef USE_MLU

#include "dragon/operators/vision/conv_transpose_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLConvTransposeOp<Context>::DoRunWithType() {
  ConvOpBase<Context>::Reshape();
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  INITIALIZE_TENSOR_VIA_SPEC(W, w_shape_, T);
  if (HasBias()) INITIALIZE_TENSOR_VIA_SPEC(Input(2), b_shape_, T);

  Y_impl_.Setup<T>(
      pads_begin_,
      pads_end_,
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
    math::Add(
        2,
        vec64_t({Y->count() / out_channels_, out_channels_}).data(),
        1,
        vec64_t({out_channels_}).data(),
        Y->template data<T, Context>(),
        Input(2).template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void CNNLConvTransposeGradientOp<Context>::DoRunWithType() {
  ConvOpBase<Context>::Reshape(true);
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  auto *dX = Output(0), *dW = Output(1);

  if (HasBias()) {
    vec64_t B_bcast_axes(dY.ndim());
    std::iota(B_bcast_axes.begin(), B_bcast_axes.end(), 0);
    if (data_format() == "NCHW") {
      B_bcast_axes.erase(B_bcast_axes.begin() + 1);
    } else if (data_format() == "NHWC") {
      B_bcast_axes.erase(B_bcast_axes.end() - 1);
    }
    dB_impl_.Setup<T>(dY.dims(), B_bcast_axes, ctx());
  }

  if (dX->has_name()) {
    dX_impl_.Setup<T>(
        pads_begin_,
        pads_end_,
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
        pads_end_,
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

  if (dW->has_name()) {
    dW_impl_.Compute<T>(
        X.template data<T, Context>(),
        dY.template data<T, Context>(),
        dW->template mutable_data<T, Context>(),
        ctx());
  }

  if (HasBias()) {
    dB_impl_.Compute<T>(
        dY.template data<T, Context>(),
        Output(2)->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(dB_impl_.scratch_size()),
        ctx());
  }
}

DEPLOY_CNNL_OPERATOR(ConvTranspose);
DEPLOY_CNNL_OPERATOR(ConvTransposeGradient);

} // namespace dragon

#endif // USE_MLU
