#include "operators/common/flatten_op.h"

namespace dragon {

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
    vector<TIndex> output_dims;
    for (int i = 0; i < axis; i++) 
        output_dims.push_back(input(0).dim(i));
    if (num_axes < 1) {
        output_dims.push_back(input(0).count(axis));
    } else {
        TIndex count = input(0).count(axis, axis + num_axes);
        output_dims.push_back(count);
        for (int i = axis + num_axes; i < input(0).ndim(); i++)
            output_dims.push_back(input(0).dim(i));
    }
    output(0)->Reshape(output_dims);
    output(0)->Share(input(0));
}

DEPLOY_CPU(Flatten);
#ifdef WITH_CUDA
DEPLOY_CUDA(Flatten);
#endif
OPERATOR_SCHEMA(Flatten).NumInputs(1).NumOutputs(1);


template <class Context>
void FlattenGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    output(0)->Share(input(1));
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