#include "operators/recurrent/recurrent_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

DEPLOY_CPU(Recurrent);
#ifdef WITH_CUDA
DEPLOY_CUDA(Recurrent);
#endif
OPERATOR_SCHEMA(Recurrent).NumInputs(4).NumOutputs(1, 3);

DEPLOY_CPU(RecurrentGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RecurrentGradient);
#endif
OPERATOR_SCHEMA(RecurrentGradient).NumInputs(6, 8).NumOutputs(4);

class GetRecurrentGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetRecurrentGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs({ I(0), I(1), I(2), I(3), O(0), GO(0) });
        if (def.output_size() > 1) inputs.push_back(GO(1));
        if (def.output_size() > 2) inputs.push_back(GO(2));
        return SingleDef(def.type() + "Gradient", "", inputs,
                vector<string> {GI(0), GI(1), GI(2), GI(3)});
    }
};
REGISTER_GRADIENT(Recurrent, GetRecurrentGradient);

}    // namespace dragon