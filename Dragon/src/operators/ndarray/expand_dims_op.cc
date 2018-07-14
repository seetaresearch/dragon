#include "core/workspace.h"
#include "operators/ndarray/expand_dims_op.h"

namespace dragon {

template <class Context>
void ExpandDimsOp<Context>::RunOnDevice() {
    vector<TIndex> dims = Input(0).dims();
    if (axis == -1 || axis >= (int)dims.size()) dims.push_back(1);
    else dims.insert(dims.begin() + axis, 1);
    //  save Xshape
    Tensor* sv = ws()->CreateTensor(
        "/mnt/" + anchor() + "/expand_dims/x_shape");
    sv->Reshape({ (TIndex)Input(0).ndim() });
    auto* Sdata = sv->template mutable_data<TIndex, CPUContext>();
    for (int i = 0; i < Input(0).ndim(); i++) Sdata[i] = Input(0).dim(i);
    Output(0)->Reshape(dims);
    if (Output(0)->name() != Input(0).name())
        Output(0)->template Copy<Context, Context>(Input(0));
}

DEPLOY_CPU(ExpandDims);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDims);
#endif
OPERATOR_SCHEMA(ExpandDims)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context>
void ExpandDimsGradientOp<Context>::RunOnDevice() {
    Tensor* sv = ws()->GetTensor(
        "/mnt/" + anchor() + "/expand_dims/x_shape");
    auto* Sdata = sv->template mutable_data<TIndex, CPUContext>();
    vector<TIndex> x_shape(sv->count());
    for (int i = 0; i < sv->count(); i++) x_shape[i] = Sdata[i]; 
    Output(0)->Reshape(x_shape);
    if (Output(0)->name() != Input(-1).name())
        Output(0)->template Copy<Context, Context>(Input(-1));
}

DEPLOY_CPU(ExpandDimsGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDimsGradient);
#endif
OPERATOR_SCHEMA(ExpandDimsGradient)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });

class GetExpandDimsGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetExpandDimsGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(ExpandDims, GetExpandDimsGradient);

}    // namespace dragon