#include "core/workspace.h"
#include "operators/array/dimension_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() \
        << ", " << X.ndim() << "), got " \
        << OpArg<int64_t>("axis", 0) << ".";

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    vec64_t out_shape;
    if (keep_axes_ != INT_MAX) {
        CHECK_LE(keep_axes_, X(0).ndim())
            << "\nThe total number of axes is "
            << X(0).ndim() << ", can not keep "
            << keep_axes_ << " .";
        int i = 0;
        for (; i < keep_axes_ - 1; i++)
            out_shape.push_back(X(0).dim(i));
        if (X(0).count(i) != 1)
            out_shape.push_back(X(0).count(i));
    } else {
        for (int i = 0; i < axis_; i++)
            out_shape.push_back(X(0).dim(i));
        if (num_axes_ < 1) {
            out_shape.push_back(X(0).count(axis_));
        } else {
            auto to = axis_ + num_axes_;
            out_shape.push_back(X(0).count(axis_, to));
            for (int i = to; i < X(0).ndim(); i++)
                out_shape.push_back(X(0).dim(i));
        }
    }
    Y(0)->Reshape(out_shape);
    Y(0)->SetMeta(X(0).meta());
    Y(0)->Share(X(0).memory());
}

DEPLOY_CPU(Flatten);
#ifdef WITH_CUDA
DEPLOY_CUDA(Flatten);
#endif

DEPLOY_CPU(FlattenGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FlattenGradient);
#endif

OPERATOR_SCHEMA(Flatten)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(FlattenGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Flatten, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon