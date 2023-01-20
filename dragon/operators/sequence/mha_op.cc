#include "dragon/operators/sequence/mha_op.h"

namespace dragon {

OPERATOR_SCHEMA(MultiHeadSelfAttn).NumInputs(2, 3).NumOutputs(1);
OPERATOR_SCHEMA(MultiHeadSelfAttnGradient).NumInputs(4).NumOutputs(2);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), I(2), GO(0)}),
        vector<string>({GI(0), GI(1)}));
  }
};

} // namespace

REGISTER_GRADIENT(MultiHeadSelfAttn, GradientMaker);

} // namespace dragon
