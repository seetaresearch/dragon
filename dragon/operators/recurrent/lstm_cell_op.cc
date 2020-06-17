#include "dragon/operators/recurrent/lstm_cell_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void LSTMCellOp<Context>::DoRunWithType() {
  auto* x = Input(0).template mutable_data<T, Context>();
  auto* hx = Input(1).template data<T, Context>();
  auto* h = Output(0)->template mutable_data<T, Context>();
  auto* c = Output(1)->template mutable_data<T, Context>();
  kernel::LSTMCell(Input(1).dim(0), Input(1).dim(-1), hx, x, c, h, ctx());
}

template <class Context>
void LSTMCellOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(1));
  Output(1)->ReshapeLike(Input(1));

  if (XIsType(Input(0), float)) {
    DoRunWithType<float>();
  } else {
    LOG(FATAL) << TypeString(Input(0), {"float32"});
  }
}

template <class Context>
template <typename T>
void LSTMCellGradientOp<Context>::DoRunWithType() {
  auto* x = Input(0).template data<T, Context>();
  auto* hx = Input(1).template data<T, Context>();
  auto* c = Input(2).template data<T, Context>();
  auto* dh = Input(3).template data<T, Context>();
  auto* dc = Input(4).template mutable_data<T, Context>();
  auto* dx = Output(0)->template mutable_data<T, Context>();
  auto* dhx = Output(1)->template mutable_data<T, Context>();

  if (!Input(-1).has_name()) {
    math::Set(Input(-1).count(), cast::to<T>(0.f), dc, ctx());
  }

  kernel::LSTMCellGrad(
      Input(1).dim(0), Input(1).dim(-1), hx, x, c, dc, dh, dhx, dx, ctx());
}

template <class Context>
void LSTMCellGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));
  Output(1)->ReshapeLike(Input(1));

  if (!Input(-1).has_name()) {
    // dC will be ignored if C is not solved
    // We should Zero-Reset the dC
    Input(-1).ReshapeLike(Input(-2));
  }

  if (XIsType(Input(0), float)) {
    DoRunWithType<float>();
  } else {
    LOG(FATAL) << TypeString(Input(0), {"float32"});
  }
}

DEPLOY_CPU(LSTMCell);
#ifdef USE_CUDA
DEPLOY_CUDA(LSTMCell);
#endif

DEPLOY_CPU(LSTMCellGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(LSTMCellGradient);
#endif

OPERATOR_SCHEMA(LSTMCell)
    /* X, HX */
    .NumInputs(2, 3)
    /* H, C */
    .NumOutputs(2);

OPERATOR_SCHEMA(LSTMCellGradient)
    /* X, HX, C, dH, dC */
    .NumInputs(5)
    /* dX, dHX */
    .NumOutputs(2);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), O(1), GO(0), GO(1)}),
        vector<string>({GI(0), GI(1)}));
  }
  vector<float> defaults() override {
    // Fill zero for dCNext
    return {1.f, 0.f};
  }
};

} // namespace

REGISTER_GRADIENT(LSTMCell, GradientMaker);

} // namespace dragon
