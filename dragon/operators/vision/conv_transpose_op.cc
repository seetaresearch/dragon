#include "dragon/operators/vision/conv_transpose_op.h"
#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_op_impl.h"

namespace dragon {

template <class Context>
template <typename T>
void ConvTransposeOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  ConvOpBase<Context>::Reshape();
  INITIALIZE_TENSOR_VIA_SPEC(W, w_shape_, T);
  if (HasBias()) INITIALIZE_TENSOR_VIA_SPEC(Input(2), b_shape_, T);

  auto* x = X.template data<T, Context>();
  auto* w = W.template data<T, Context>();
  auto* y = Y->template mutable_data<T, Context>();

  for (int i = 0; i < X.dim(0); ++i) {
    BwdData(x + i * X_stride_, w, y + i * Y_stride_);
  }

  if (HasBias()) {
    FwdBias(Input(2).template data<T, Context>(), y);
  }
}

template <class Context>
template <typename T>
void ConvTransposeGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  ConvOpBase<Context>::Reshape(true);

  auto* dy = dY.template data<T, Context>();

  if (dX->has_name()) {
    auto* w = W.template data<T, Context>();
    auto* dx = dX->template mutable_data<T, Context>();
    for (int i = 0; i < X.dim(0); ++i) {
      FwdData(dy + i * Y_stride_, w, dx + i * X_stride_);
    }
  }

  if (dW->has_name()) {
    auto* x = X.template data<T, Context>();
    auto* dy = dY.template data<T, Context>();
    auto* dw = dW->template mutable_data<T, Context>();
    for (int i = 0; i < X.dim(0); ++i) {
      BwdFilter(x + i * X_stride_, dy + i * Y_stride_, dw, i > 0);
    }
  }

  if (dB->has_name()) {
    BwdBias(dy, dB->template mutable_data<T, Context>());
  }
}

DEPLOY_CPU_OPERATOR(ConvTranspose);
DEPLOY_CPU_OPERATOR(ConvTransposeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ConvTranspose);
DEPLOY_CUDA_OPERATOR(ConvTransposeGradient);
#endif

OPERATOR_SCHEMA(ConvTranspose)
    /* X, W, B */
    .NumInputs(2, 3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ConvTransposeGradient)
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

REGISTER_GRADIENT(ConvTranspose, GradientMaker);

} // namespace dragon
