#include "core/workspace.h"
#include "operators/array/dimension_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", INT_MAX); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0) \
        << "\nExcepted the axis in [-" << X.ndim() \
        << ", INT_MAX), got " \
        << OpArg<int64_t>("axis", 0) << ".";

template <class Context>
void SqueezeOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    vec64_t out_shape;
    for (int i = 0; i < X(0).ndim(); i++) {
        if ((X(0).dim(i) != 1) ||
                (axis_ != INT_MAX &&
                    X(0).dim(i) == 1 &&
                        i != axis_)) {
            out_shape.push_back(X(0).dim(i));
        }
    }

    Y(0)->Reshape(out_shape);
    Y(0)->SetMeta(X(0).meta());
    Y(0)->Share(X(0).memory());
}

DEPLOY_CPU(Squeeze);
#ifdef WITH_CUDA
DEPLOY_CUDA(Squeeze);
#endif

DEPLOY_CPU(SqueezeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SqueezeGradient);
#endif

OPERATOR_SCHEMA(Squeeze)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SqueezeGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Squeeze, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon