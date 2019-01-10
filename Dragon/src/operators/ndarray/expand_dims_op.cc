#include "core/workspace.h"
#include "operators/ndarray/dimension_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 0); \
    axis = axis < 0 ? axis + X.ndim() + 1 : axis; \
    CHECK(axis >= 0 && axis <= X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() + 1 << ", " << X.ndim() \
       << "], got " << OperatorBase::Arg<int64_t>("axis", 0) << ".";

template <class Context>
void ExpandDimsOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));
    vector<int64_t> dims = Input(0).dims();
    dims.insert(dims.begin() + axis, 1);

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
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(ExpandDims, SimpleGradientMaker);

}  // namespace dragon