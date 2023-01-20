#include "dragon/operators/vision/depthwise_conv_op.h"
#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_op_impl.h"

namespace dragon {

template <class Context>
template <typename T>
void DepthwiseConvOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  group_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  ConvOpBase<Context>::Reshape();
  INITIALIZE_TENSOR_VIA_SPEC(W, w_shape_, T);
  if (HasBias()) INITIALIZE_TENSOR_VIA_SPEC(Input(2), b_shape_, T);

  CHECK_EQ(in_channels_, out_channels_)
      << "\nInput and output channels to be same.";

  if (num_axes_ == 1 || num_axes_ == 2) {
    kernels::DepthwiseConv2d(
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
    FwdBias(
        Input(2).template data<T, Context>(),
        Y->template mutable_data<T, Context>());
  }
}

template <class Context>
template <typename T>
void DepthwiseConvGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);

  group_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  ConvOpBase<Context>::Reshape(true);

  if (dX->has_name()) {
    if (num_axes_ == 1 || num_axes_ == 2) {
      kernels::DepthwiseConv2dGrad(
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
      kernels::DepthwiseConv2dWGrad(
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
    BwdBias(
        dY.template data<T, Context>(),
        dB->template mutable_data<T, Context>());
  }
}

DEPLOY_CPU_OPERATOR(DepthwiseConv);
DEPLOY_CPU_OPERATOR(DepthwiseConvGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DepthwiseConv);
DEPLOY_CUDA_OPERATOR(DepthwiseConvGradient);
#endif

OPERATOR_SCHEMA(DepthwiseConv)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(DepthwiseConvGradient)
    /* X, W, dY */
    .NumInputs(3)
    /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(DepthwiseConv, GradientMaker);

} // namespace dragon
