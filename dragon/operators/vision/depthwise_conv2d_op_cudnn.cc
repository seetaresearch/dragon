#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/depthwise_conv_op.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNDepthwiseConv2dOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);

  group_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  ConvOpBase<Context>::Reshape();
  CHECK_EQ(in_channels_, out_channels_)
      << "\nExcepted in/out channels to be same.";

  TENSOR_FILL(W, w_shape_);
  kernel::DepthwiseConv2d(
      X.dim(0),
      in_channels_,
      in_shape_[0],
      in_shape_[1],
      out_shape_[0],
      out_shape_[1],
      kshape_[0],
      kshape_[1],
      stride_[0],
      stride_[1],
      pad_l_[0],
      pad_l_[1],
      dilation_[0],
      dilation_[1],
      data_format(),
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());

  if (HasBias()) {
    TENSOR_FILL(Input(2), b_shape_);
    CuDNNSetBiasDesc<T>(&bias_desc_, 4, out_channels_, data_format());
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
void CuDNNDepthwiseConv2dOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNDepthwiseConv2dGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);

  group_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  ConvOpBase<Context>::Reshape(true);

  if (dX->has_name()) {
    kernel::DepthwiseConv2dGrad(
        X.dim(0),
        in_channels_,
        in_shape_[0],
        in_shape_[1],
        out_shape_[0],
        out_shape_[1],
        kshape_[0],
        kshape_[1],
        stride_[0],
        stride_[1],
        pad_l_[0],
        pad_l_[1],
        dilation_[0],
        dilation_[1],
        data_format(),
        dY.template data<T, Context>(),
        W.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }

  if (dW->has_name()) {
    kernel::DepthwiseConv2dWGrad(
        X.dim(0),
        in_channels_,
        in_shape_[0],
        in_shape_[1],
        out_shape_[0],
        out_shape_[1],
        kshape_[0],
        kshape_[1],
        stride_[0],
        stride_[1],
        pad_l_[0],
        pad_l_[1],
        dilation_[0],
        dilation_[1],
        data_format(),
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dW->template mutable_data<T, Context>(),
        ctx());
  }

  if (dB->has_name()) {
    CuDNNSetTensorDesc<T>(&input_desc_, Input(-1).dims(), data_format());
    CuDNNSetBiasDesc<T>(&bias_desc_, 4, out_channels_, data_format());
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
void CuDNNDepthwiseConv2dGradientOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float>>::Call(this, Input(0));
}

DEPLOY_CUDNN(DepthwiseConv2d);
DEPLOY_CUDNN(DepthwiseConv2dGradient);

} // namespace dragon

#endif // USE_CUDNN
