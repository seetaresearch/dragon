#include "core/workspace.h"
#include "operators/array/dimension_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 0); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 0) << ".";

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    vector<int64_t> output_dims;
    if (keep_axes != INT_MAX) {
        CHECK_LE(keep_axes, Input(0).ndim())
            << "\nThe total number of axes is " + Input(0).ndim()
            << ", can not keep " + keep_axes << " .";
        int i = 0;
        for (; i < keep_axes - 1; i++)
            output_dims.push_back(Input(0).dim(i));
        if (Input(0).count(i) != 1)
            output_dims.push_back(Input(0).count(i));
    } else {
        for (int i = 0; i < axis; i++)
            output_dims.push_back(Input(0).dim(i));
        if (num_axes < 1) {
            output_dims.push_back(Input(0).count(axis));
        } else {
            int64_t count = Input(0).count(axis, axis + num_axes);
            output_dims.push_back(count);
            for (int i = axis + num_axes; i < Input(0).ndim(); i++)
                output_dims.push_back(Input(0).dim(i));
        }
    }
    Output(0)->Reshape(output_dims);
    Output(0)->SetMeta(Input(0).meta());
    Output(0)->Share(Input(0).memory());
}

DEPLOY_CPU(Flatten);
#ifdef WITH_CUDA
DEPLOY_CUDA(Flatten);
#endif
OPERATOR_SCHEMA(Flatten).NumInputs(1).NumOutputs(1);


DEPLOY_CPU(FlattenGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FlattenGradient);
#endif

OPERATOR_SCHEMA(FlattenGradient)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Flatten, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon