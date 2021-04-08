#include "dragon/operators/recurrent/recurrent_op.h"

namespace dragon {

DEPLOY_CPU_OPERATOR(Recurrent);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Recurrent);
#endif

DEPLOY_CPU_OPERATOR(RecurrentGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RecurrentGradient);
#endif

OPERATOR_SCHEMA(Recurrent)
    /* X, W, HX, CX */
    .NumInputs(2, 4)
    /* Y, HY, CY */
    .NumOutputs(1, 3);

OPERATOR_SCHEMA(RecurrentGradient)
    /* X, W, HX, CX, Y, dY, dHY, dCY */
    .NumInputs(6, 8)
    /* dX, dW, dHX, dCX */
    .NumOutputs(4);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    vector<string> inputs({
        I(0),
        I(1),
        def().input_size() > 2 ? I(2) : "",
        def().input_size() > 3 ? I(3) : "",
        O(0),
        GO(0),
        def().output_size() > 1 ? GO(1) : "",
        def().output_size() > 2 ? GO(2) : "",
    });
    vector<string> outputs({
        GI(0),
        GI(1),
        def().input_size() > 2 ? GI(2) : "",
        def().input_size() > 3 ? GI(3) : "",
    });
    AddGradientDef(def().type() + "Gradient", "", inputs, outputs);
  }
};

} // namespace

REGISTER_GRADIENT(Recurrent, GradientMaker);

} // namespace dragon
