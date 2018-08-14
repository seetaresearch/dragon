#include "core/workspace.h"
#include "operators/ndarray/dimension_op.h"

namespace dragon {

template <class Context>
void ExpandDimsOp<Context>::RunOnDevice() {
    TIndex _axis_ = axis >= 0 ? axis :
        axis + (TIndex)Input(0).ndim() + 1;
    vector<TIndex> dims = Input(0).dims();
    if (_axis_ < 0 ||
            _axis_ >= (TIndex)dims.size())
                dims.push_back(1);
    else dims.insert(dims.begin() + _axis_, 1);
    Output(0)->Reshape(dims);
    Output(0)->SetMeta(Input(0).meta());
    Output(0)->Share(Input(0).memory());
}

DEPLOY_CPU(ExpandDims);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDims);
#endif
OPERATOR_SCHEMA(ExpandDims).NumInputs(1).NumOutputs(1);


DEPLOY_CPU(ExpandDimsGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDimsGradient);
#endif
OPERATOR_SCHEMA(ExpandDimsGradient)
    .NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 } });

class GetExpandDimsGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetExpandDimsGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(ExpandDims, GetExpandDimsGradient);

}    // namespace dragon