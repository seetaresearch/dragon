#include "dragon/operators/normalization/layer_norm_op.h"

namespace dragon {

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
