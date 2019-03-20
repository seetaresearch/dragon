#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/recurrent/recurrent_op.h"

namespace dragon {

DEPLOY_CPU(Recurrent);
#ifdef WITH_CUDA
DEPLOY_CUDA(Recurrent);
#endif
OPERATOR_SCHEMA(Recurrent).NumInputs(2, 4).NumOutputs(1, 3);

DEPLOY_CPU(RecurrentGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RecurrentGradient);
#endif

OPERATOR_SCHEMA(RecurrentGradient)
    .NumInputs(6, 8).NumOutputs(4);

class GetRecurrentGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetRecurrentGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs({ I(0), I(1),
            def.input_size() > 2 ? I(2) : "NULL",
            def.input_size() > 3 ? I(3) : "NULL",
            O(0), GO(0),
            def.output_size() > 1 ? GO(1) : "NULL",
            def.output_size() > 2 ? GO(2) : "NULL"});
        vector<string> outputs({ GI(0), GI(1),
            def.input_size() > 2 ? GI(2) : "NULL",
            def.input_size() > 3 ? GI(3) : "NULL"});
        return SingleDef(def.type() + "Gradient", "", inputs, outputs);
    }
};

REGISTER_GRADIENT(Recurrent, GetRecurrentGradient);

}  // namespace dragon