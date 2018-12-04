#include "core/workspace.h"
#include "operators/ndarray/dimension_op.h"

namespace dragon {

template <class Context>
void SqueezeOp<Context>::RunOnDevice() {
    TIndex _axis_ = axis >= 0 ? axis :
        axis + (TIndex)Input(0).ndim();
    vector<TIndex> dims;
    for (int i = 0; i < Input(0).ndim(); i++)
        if ((Input(0).dim(i) != 1) ||
                (_axis_ != INT_MAX &&
                    Input(0).dim(i) == 1 &&
                        i != _axis_))
                            dims.push_back(Input(0).dim(i));
    Output(0)->Reshape(dims);
    Output(0)->SetMeta(Input(0).meta());
    Output(0)->Share(Input(0).memory());
}

DEPLOY_CPU(Squeeze);
#ifdef WITH_CUDA
DEPLOY_CUDA(Squeeze);
#endif
OPERATOR_SCHEMA(Squeeze).NumInputs(1).NumOutputs(1);


DEPLOY_CPU(SqueezeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SqueezeGradient);
#endif
OPERATOR_SCHEMA(SqueezeGradient)
    .NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 } });

class GetSqueezeGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSqueezeGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Squeeze, GetSqueezeGradient);

}  // namespace dragon