#include "operators/norm/batch_norm_op.h"
#include "core/workspace.h"

namespace dragon {

DEPLOY_CPU(FusedBatchNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedBatchNorm);
#endif
OPERATOR_SCHEMA(FusedBatchNorm).NumInputs(5).NumOutputs(1);

template <class Context>
void FusedBatchNormGradientOp<Context>::ShareGradient() {
    if (use_global_stats) {
        if (output(0)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer("Grad");
            ws()->CreateAvatar(output(0), dX);
        }
    } else {
        if (output(0)->name() != "ignore" ||
            output(1)->name() != "ignore" ||
            output(2)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer("Grad");
            ws()->CreateAvatar(output(0), dX);
        }
    }
}

DEPLOY_CPU(FusedBatchNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedBatchNormGradient);
#endif
OPERATOR_SCHEMA(FusedBatchNormGradient).NumInputs(5).NumOutputs(3);

class GetFusedBatchNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetFusedBatchNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), I(2), I(3), GO(0)},
            vector<string> {GI(0), GI(3), GI(4)});
    }
};
REGISTER_GRADIENT(FusedBatchNorm, GetFusedBatchNormGradient);

}    // namespace dragon