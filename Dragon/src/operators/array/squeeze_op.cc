#include "core/workspace.h"
#include "operators/array/dimension_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", INT_MAX); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0) \
        << "\nExcepted the axis in [-" << X.ndim() << ", INT_MAX), " \
        << "got " << OperatorBase::Arg<int64_t>("axis", 0) << ".";

template <class Context>
void SqueezeOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    vector<int64_t> dims;
    for (int i = 0; i < Input(0).ndim(); i++) {
        if ((Input(0).dim(i) != 1) ||
                (axis != INT_MAX &&
                    Input(0).dim(i) == 1 &&
                        i != axis)) {
            dims.push_back(Input(0).dim(i));
        }
    }

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
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Squeeze, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon