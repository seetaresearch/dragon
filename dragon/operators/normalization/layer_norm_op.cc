#include "dragon/operators/normalization/layer_norm_op.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void LayerNormOp<Context>::DoRunWithType() {
  using AccT = typename math::Traits<T>::accumulator_type;
  auto &X = Input(0), *Y = Output(0);
  auto &W = Input(1), &B = Input(2);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto N = X.count(0, axis);
  const auto C = X.count(axis);
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({C}), AccT);
  INITIALIZE_TENSOR_VIA_SPEC(B, vec64_t({C}), AccT);
  auto* X_mu = Output("X_mu")->Reshape({N});
  auto* X_rsig = Output("X_rsig")->Reshape({N});

  kernels::LayerNorm(
      N,
      C,
      epsilon_,
      X.template data<T, Context>(),
      W.template data<AccT, Context>(),
      B.template data<AccT, Context>(),
      X_mu->template mutable_data<AccT, Context>(),
      X_rsig->template mutable_data<AccT, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(LayerNorm);
DEPLOY_CPU_OPERATOR(LayerNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LayerNorm);
DEPLOY_CUDA_OPERATOR(LayerNormGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(LayerNorm, LayerNorm);
DEPLOY_MPS_OPERATOR(LayerNormGradient, LayerNormGradient);
#endif

OPERATOR_SCHEMA(LayerNorm)
    /* X, W, B */
    .NumInputs(3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(LayerNormGradient)
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

REGISTER_GRADIENT(LayerNorm, GradientMaker);

} // namespace dragon
