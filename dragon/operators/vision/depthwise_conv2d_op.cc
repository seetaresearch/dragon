#include "dragon/core/workspace.h"
#include "dragon/operators/vision/depthwise_conv_op.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void DepthwiseConv2dOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);

  group_ = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  ConvOpBase<Context>::Reshape();
  CHECK_EQ(in_channels_, out_channels_)
      << "\nExcepted in/out channels to be same.";

  TENSOR_FILL(W, w_shape_);
  kernel::DepthwiseConv2d(
      Input(0).dim(0),
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
    Pb(Input(2).template data<T, Context>(),
       Y->template mutable_data<T, Context>());
  }
}

template <class Context>
void DepthwiseConv2dOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void DepthwiseConv2dGradientOp<Context>::DoRunWithType() {
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
    Db(dY.template data<T, Context>(), dB->template mutable_data<T, Context>());
  }
}

template <class Context>
void DepthwiseConv2dGradientOp<Context>::RunOnDevice() {
  DispatchHelper<TensorTypes<float>>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(DepthwiseConv2d);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DepthwiseConv2d);
#endif

DEPLOY_CPU_OPERATOR(DepthwiseConv2dGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DepthwiseConv2dGradient);
#endif

OPERATOR_SCHEMA(DepthwiseConv2d)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(DepthwiseConv2dGradient)
    /* X, W, dY */
    .NumInputs(3)
    /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(DepthwiseConv2d, GradientMaker);

} // namespace dragon
