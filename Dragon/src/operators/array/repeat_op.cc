#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/repeat_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", INT_MAX); \
    if (axis_ != INT_MAX) { \
        axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
        CHECK(axis_ >= 0 && axis_ < X.ndim()) \
            << "\nExcepted the axis in [-" << X.ndim() \
            << ", " << X.ndim() << "), got " \
            << OpArg<int64_t>("axis", 0) << "."; \
    }

template <class Context> template <typename T>
void RepeatOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Repeat(
        outer_dim_,
        inner_dim_,
        axis_dim_,
        repeats(),
        x, y, ctx()
    );
}

template <class Context>
void RepeatOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    if (axis_ == INT_MAX) {
        outer_dim_ = inner_dim_ = 1;
        axis_dim_  = X(0).count();
        Y(0)->Reshape({ axis_dim_ * repeats() });
    } else {
        axis_dim_  = X(0).dim(axis_);
        outer_dim_ = X(0).count(0, axis_);
        inner_dim_ = X(0).count(axis_ + 1);
        auto out_shape = X(0).dims();
        out_shape[axis_] *= repeats();
        Y(0)->Reshape(out_shape);
    }

    if (XIsType(X(0), bool)) {
        RunImpl<bool>();
    } else if (XIsType(X(0), int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(X(0), uint8_t)) {
        RunImpl<uint8_t>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), double)) {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "bool", "int8", "uint8", "int32", "int64",
                 "float16", "float32", "float64",
        });
    }
}

template <class Context> template <typename T>
void RepeatGradientOp<Context>::RunImpl() {
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::RepeatGrad(
        outer_dim_,
        inner_dim_,
        axis_dim_,
        repeats(),
        dy, dx, ctx()
    );
}

template <class Context>
void RepeatGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    if (axis_ == INT_MAX) {
        outer_dim_ = inner_dim_ = 1;
        axis_dim_  = X(0).count();
    } else {
        axis_dim_  = X(0).dim(axis_);
        outer_dim_ = X(0).count(0, axis_);
        inner_dim_ = X(0).count(axis_ + 1);
    }

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(X(0), uint8_t)) {
        RunImpl<uint8_t>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), double)) {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Repeat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Repeat);
#endif

DEPLOY_CPU(RepeatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RepeatGradient);
#endif

OPERATOR_SCHEMA(Repeat)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(RepeatGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Repeat, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon