#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_op_impl.h"
#include "dragon/operators/vision/depthwise_conv_op.h"
#include "dragon/utils/filler.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNDepthwiseConvOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);

  group_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  ConvOpBase<Context>::Reshape();
  CHECK_EQ(in_channels_, out_channels_)
      << "\nExcepted in/out channels to be same.";

  TENSOR_FILL(W, w_shape_);

  if (num_axes_ == 1 || num_axes_ == 2) {
    kernel::DepthwiseConv2d(
        X.dim(0),
        in_channels_,
        in_shape_[0],
        num_axes_ == 1 ? 1 : in_shape_[1],
        out_shape_[0],
        num_axes_ == 1 ? 1 : out_shape_[1],
        kshape_[0],
        num_axes_ == 1 ? 1 : kshape_[1],
        strides_[0],
        num_axes_ == 1 ? 1 : strides_[1],
        pads_begin_[0],
        num_axes_ == 1 ? 0 : pads_begin_[1],
        dilations_[0],
        num_axes_ == 1 ? 1 : dilations_[1],
        data_format(),
        X.template data<T, Context>(),
        W.template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "DepthwiseConv" << num_axes_ << "d is not supported";
  }

  if (HasBias()) {
    TENSOR_FILL(Input(2), b_shape_);
    CuDNNSetBiasDesc<T>(&bias_desc_, X.ndim(), out_channels_, data_format());
    CuDNNSetTensorDesc<T>(&output_desc_, Y->dims(), data_format());
    CUDNN_CHECK(cudnnAddTensor(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        bias_desc_,
        Input(2).template data<T, Context>(),
        CuDNNType<T>::one,
        output_desc_,
        Y->template mutable_data<T, Context>()));
  }
}

template <class Context>
void CuDNNDepthwiseConvOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float16, float>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNDepthwiseConvGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);

  group_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  ConvOpBase<Context>::Reshape(true);

  if (dX->has_name()) {
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernel::DepthwiseConv2dGrad(
          X.dim(0),
          in_channels_,
          in_shape_[0],
          num_axes_ == 1 ? 1 : in_shape_[1],
          out_shape_[0],
          num_axes_ == 1 ? 1 : out_shape_[1],
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          dilations_[0],
          num_axes_ == 1 ? 1 : dilations_[1],
          data_format(),
          dY.template data<T, Context>(),
          W.template data<T, Context>(),
          dX->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "DepthwiseConv" << num_axes_ << "d is not supported";
    }
  }

  if (dW->has_name()) {
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernel::DepthwiseConv2dWGrad(
          X.dim(0),
          in_channels_,
          in_shape_[0],
          num_axes_ == 1 ? 1 : in_shape_[1],
          out_shape_[0],
          num_axes_ == 1 ? 1 : out_shape_[1],
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          dilations_[0],
          num_axes_ == 1 ? 1 : dilations_[1],
          data_format(),
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          dW->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "DepthwiseConv" << num_axes_ << "d is not supported";
    }
  }

  if (dB->has_name()) {
    CuDNNSetTensorDesc<T>(&input_desc_, dY.dims(), data_format());
    CuDNNSetBiasDesc<T>(&bias_desc_, dY.ndim(), out_channels_, data_format());
    CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        input_desc_,
        dY.template data<T, Context>(),
        CuDNNType<T>::zero,
        bias_desc_,
        dB->template mutable_data<T, Context>()));
  }
}

template <class Context>
void CuDNNDepthwiseConvGradientOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float16, float>>::Call(this, Input(0));
}

DEPLOY_CUDNN_OPERATOR(DepthwiseConv);
DEPLOY_CUDNN_OPERATOR(DepthwiseConvGradient);

} // namespace dragon

#endif // USE_CUDNN
