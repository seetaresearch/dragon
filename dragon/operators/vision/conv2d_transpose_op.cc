#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_transpose_op.h"
#include "dragon/utils/filler.h"

namespace dragon {

template <class Context>
template <typename T>
void ConvTranspose2dOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);

  ConvOpBase<Context>::Reshape();
  // Fix the output shape for im2col/col2im
  for (int i = 0; i < num_axes_; i++) {
    out_shape_[i] = X.dim(axis_ + i);
  }

  TENSOR_FILL(W, w_shape_);
  auto* x = X.template data<T, Context>();
  auto* w = W.template data<T, Context>();
  auto* y = Y->template mutable_data<T, Context>();

  for (int i = 0; i < X.dim(0); ++i) {
    Dx(x + i * x_ofs_, w, y + i * y_ofs_);
  }

  if (HasBias()) {
    TENSOR_FILL(Input(2), b_shape_);
    Pb(Input(2).template data<T, Context>(), y);
  }
}

template <class Context>
void ConvTranspose2dOp<Context>::RunOnDevice() {
  if (data_format() == "NHWC" && group_ != 1) {
    // You really need the CuDNN to help you -:)
    LOG(FATAL) << "GroupConv(NHWC) is not supported.";
  }
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ConvTranspose2dGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);

  ConvOpBase<Context>::Reshape(true);
  // Fix the output shape for im2col/col2im
  for (int i = 0; i < num_axes_; i++) {
    out_shape_[i] = X.dim(axis_ + i);
  }

  if (dX->has_name()) {
    auto* dy = dY.template data<T, Context>();
    auto* w = W.template data<T, Context>();
    auto* dx = dX->template mutable_data<T, Context>();
    for (int i = 0; i < X.dim(0); ++i) {
      Wx(dy + i * y_ofs_, w, dx + i * x_ofs_);
    }
  }

  if (dW->has_name()) {
    auto* x = X.template data<T, Context>();
    auto* dy = dY.template data<T, Context>();
    auto* dw = dW->template mutable_data<T, Context>();
    for (int i = 0; i < X.dim(0); ++i) {
      Dw(x + i * x_ofs_, dy + i * y_ofs_, dw, i > 0);
    }
  }

  if (dB->has_name()) {
    Db(dY.template data<T, Context>(), dB->template mutable_data<T, Context>());
  }
}

template <class Context>
void ConvTranspose2dGradientOp<Context>::RunOnDevice() {
  if (data_format() == "NHWC" && group_ != 1) {
    // You really need the CuDNN to help you -:)
    LOG(FATAL) << "GroupConv(NHWC) is not supported.";
  }
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
}

DEPLOY_CPU(ConvTranspose2d);
#ifdef USE_CUDA
DEPLOY_CUDA(ConvTranspose2d);
#endif

DEPLOY_CPU(ConvTranspose2dGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(ConvTranspose2dGradient);
#endif

OPERATOR_SCHEMA(ConvTranspose2d)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ConvTranspose2dGradient)
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

REGISTER_GRADIENT(ConvTranspose2d, GradientMaker);

} // namespace dragon
