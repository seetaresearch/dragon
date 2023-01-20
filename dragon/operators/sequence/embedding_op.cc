#include "dragon/operators/sequence/embedding_op.h"

namespace dragon {

OPERATOR_SCHEMA(Embedding).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(EmbeddingGradient).NumInputs(2).NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(Embedding, GradientMaker);

} // namespace dragon
