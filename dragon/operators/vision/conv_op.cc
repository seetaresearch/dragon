#include "dragon/operators/vision/conv_op.h"
#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_op_impl.h"

namespace dragon {

template <class Context>
template <typename T>
void ConvOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  ConvOpBase<Context>::Reshape();
  INITIALIZE_TENSOR_VIA_SPEC(W, w_shape_, T);

  auto* x = X.template data<T, Context>();
  auto* w = W.template data<T, Context>();
  auto* y = Y->template mutable_data<T, Context>();

  for (int i = 0; i < X.dim(0); ++i) {
    WeightedX(x + i * X_stride_, w, y + i * Y_stride_);
  }

  if (HasBias()) {
    INITIALIZE_TENSOR_VIA_SPEC(Input(2), b_shape_, T);
    AddBias(Input(2).template data<T, Context>(), y);
  }
}

template <class Context>
void ConvOp<Context>::RunOnDevice() {
  if (data_format() == "NHWC" && group_ != 1) {
    // You really need the CuDNN to help you -:)
    LOG(FATAL) << "GroupConv(NHWC) is not supported.";
  }
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ConvGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  ConvOpBase<Context>::Reshape(true);

  if (dX->has_name()) {
    auto* dy = dY.template data<T, Context>();
    auto* w = W.template data<T, Context>();
    auto* dx = dX->template mutable_data<T, Context>();
    for (int i = 0; i < X.dim(0); ++i) {
      GradX(dy + i * Y_stride_, w, dx + i * X_stride_);
    }
  }

  if (dW->has_name()) {
    auto* dy = dY.template data<T, Context>();
    auto* x = X.template data<T, Context>();
    auto* dw = dW->template mutable_data<T, Context>();
    for (int i = 0; i < X.dim(0); ++i) {
      GradW(dy + i * Y_stride_, x + i * X_stride_, dw, i > 0);
    }
  }

  if (dB->has_name()) {
    GradBias(
        dY.template data<T, Context>(),
        dB->template mutable_data<T, Context>());
  }
}

template <class Context>
void ConvGradientOp<Context>::RunOnDevice() {
  if (data_format() == "NHWC" && group_ != 1) {
    // You really need the CuDNN to help you -:)
    LOG(FATAL) << "GroupConv(NHWC) is not supported.";
  }
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Conv);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Conv);
#endif

DEPLOY_CPU_OPERATOR(ConvGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ConvGradient);
#endif

OPERATOR_SCHEMA(Conv)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ConvGradient)
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

REGISTER_GRADIENT(Conv, GradientMaker);

} // namespace dragon
