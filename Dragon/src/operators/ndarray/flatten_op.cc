#include "core/workspace.h"
#include "operators/ndarray/flatten_op.h"

namespace dragon {

template <class Context>
void FlattenOp<Context>::SqueezeRun() {
    vector<TIndex> output_dims;
    for (int i = 0; i < axis; i++)
        output_dims.push_back(Input(0).dim(i));
    if (num_axes < 1) {
        output_dims.push_back(Input(0).count(axis));
    } else {
        TIndex count = Input(0).count(axis, axis + num_axes);
        output_dims.push_back(count);
        for (int i = axis + num_axes; i < Input(0).ndim(); i++)
            output_dims.push_back(Input(0).dim(i));
    }
    Output(0)->Reshape(output_dims);
    if (Output(0)->name() != Input(0).name())
        Output(0)->template Copy<Context, Context>(Input(0));
}

template <class Context>
void FlattenOp<Context>::KeepRun() {
    CHECK_LE(keep_axes, (int)Input(0).ndim())
        << "\nThe total number of axes is " + Input(0).ndim()
        << ", can not keep " + keep_axes << " .";
    vector<TIndex> output_dims;
    int i = 0;
    for (; i < keep_axes - 1; i++)
        output_dims.push_back(Input(0).dim(i));

    if (Input(0).count(i) != 1)
        output_dims.push_back(Input(0).count(i));

    if (Output(0)->name() != Input(0).name())
        Output(0)->template Copy<Context, Context>(Input(0));
}

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
    //  save Xshape
    Tensor* sv = ws()->CreateTensor(
        "/mnt/" + anchor() + "/flatten/x_shape");
    sv->Reshape({ (TIndex)Input(0).ndim() });
    auto* Sdata = sv->template mutable_data<TIndex, CPUContext>();
    for (int i = 0; i < Input(0).ndim(); i++) 
        Sdata[i] = Input(0).dim(i);
    if (keep_axes != INT_MAX) KeepRun();
    else SqueezeRun();
}

DEPLOY_CPU(Flatten);
#ifdef WITH_CUDA
DEPLOY_CUDA(Flatten);
#endif
OPERATOR_SCHEMA(Flatten)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });


template <class Context>
void FlattenGradientOp<Context>::RunOnDevice() {
    Tensor* sv = ws()->GetTensor(
        "/mnt/" + anchor() + "/flatten/x_shape");
    auto* Sdata = sv->template mutable_data<TIndex, CPUContext>();
    vector<TIndex> x_shape(sv->count());
    for (int i = 0; i < sv->count(); i++) x_shape[i] = Sdata[i];
    Output(0)->Reshape(x_shape);
    if (Output(0)->name() != Input(-1).name())
        Output(0)->template Copy<Context, Context>(Input(-1));
}

DEPLOY_CPU(FlattenGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FlattenGradient);
#endif
OPERATOR_SCHEMA(FlattenGradient)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });

class GetFlattenGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetFlattenGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Flatten, GetFlattenGradient);

} // namespace dragon