#include "dragon/operators/normalization/layer_norm_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void LayerNormOp<Context>::DoRunWithType() {
  using ParamT = typename math::AccmulatorType<T>::type;
  auto &X = Input(0), *Y = Output(0);
  auto &W = Input(1), &B = Input(2);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto N = X.count(0, axis);
  const auto C = X.count(axis);
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({C}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(B, vec64_t({C}), ParamT);
  auto* X_mu = Buffer("X_mu")->Reshape({N});
  auto* X_rsig = Buffer("X_rsig")->Reshape({N});

  kernels::LayerNorm(
      N,
      C,
      epsilon_,
      X.template data<T, Context>(),
      W.template data<ParamT, Context>(),
      B.template data<ParamT, Context>(),
      X_mu->template mutable_data<ParamT, Context>(),
      X_rsig->template mutable_data<ParamT, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(LayerNorm);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LayerNorm);
#endif

DEPLOY_CPU_OPERATOR(LayerNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LayerNormGradient);
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
