#include "operators/ndarray/expand_dims_op.h"

namespace dragon {

template <class Context>
void ExpandDimsOp<Context>::RunOnDevice() {
    vector<TIndex> dims = Input(0).dims();
    if (axis == -1 || axis >= (int)dims.size()) dims.push_back(1);
    else dims.insert(dims.begin() + axis, 1);
    Output(0)->Reshape(dims);
    Output(0)->Share(Input(0));
}

DEPLOY_CPU(ExpandDims);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDims);
#endif
OPERATOR_SCHEMA(ExpandDims).NumInputs(1).NumOutputs(1);

template <class Context>
void ExpandDimsGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    Output(0)->Share(Input(-1));
}

DEPLOY_CPU(ExpandDimsGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDimsGradient);
#endif
OPERATOR_SCHEMA(ExpandDimsGradient).NumInputs(2).NumOutputs(1);

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