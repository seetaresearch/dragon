#include "core/workspace.h"
#include "operators/array/dimension_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() + 1 : axis_; \
    CHECK(axis_ >= 0 && axis_ <= X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() + 1 \
        << ", " << X.ndim() << "], got " \
        << OpArg<int64_t>("axis", 0) << ".";

template <class Context>
void ExpandDimsOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    auto out_shape = X(0).dims();
    out_shape.insert(out_shape.begin() + axis_, 1);

    Y(0)->Reshape(out_shape);
    Y(0)->SetMeta(X(0).meta());
    Y(0)->Share(X(0).memory());
}

DEPLOY_CPU(ExpandDims);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDims);
#endif

DEPLOY_CPU(ExpandDimsGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ExpandDimsGradient);
#endif

OPERATOR_SCHEMA(ExpandDims)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ExpandDimsGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(ExpandDims, SimpleGradientMaker);

}  // namespace dragon