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
    Output(0)->Share(Input(0));
}

template <class Context>
void FlattenOp<Context>::KeepRun() {
    CHECK_LE(keep_axes, (int)Input(0).ndim())
        << "\nThe total number of axes is " + Input(0).ndim()
        << ", can not keep " + keep_axes << " .";
    vector<TIndex> output_dims;
    int i = 0;
    for (; i < keep_axes - 1; i++) output_dims.push_back(Input(0).dim(i));
    if (Input(0).count(i) != 1) output_dims.push_back(Input(0).count(i));
    Output(0)->Reshape(output_dims);
    Output(0)->Share(Input(0));
}

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
    if (keep_axes != INT_MAX) KeepRun();
    else SqueezeRun();
}

DEPLOY_CPU(Flatten);
#ifdef WITH_CUDA
DEPLOY_CUDA(Flatten);
#endif
OPERATOR_SCHEMA(Flatten).NumInputs(1).NumOutputs(1);


template <class Context>
void FlattenGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    Output(0)->Share(Input(1));
}

DEPLOY_CPU(FlattenGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FlattenGradient);
#endif
OPERATOR_SCHEMA(FlattenGradient).NumInputs(2).NumOutputs(1);

class GetFlattenGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetFlattenGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Flatten, GetFlattenGradient);

} // namespace dragon