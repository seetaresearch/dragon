#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/concat_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() \
        << ", " << X.ndim() << "), got " \
        << OpArg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void ConcatOp<Context>::RunImpl() {
    int64_t axis_dim, cat_ofs = 0;
    auto* y = Y(0)->template mutable_data<T, Context>();

    for (int i = 0; i < XSize(); i++) {
        axis_dim = X(i).dim(axis_);
        auto* x = X(i).template data<T, Context>();
        kernel::Concat(
            outer_dim_,
            inner_dim_,
            axis_dim,
            cat_dim_,
            cat_ofs,
            x, y, ctx()
        );
        cat_ofs += axis_dim;
    }
}

template <class Context>
void ConcatOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    auto out_shape = X(0).dims();

    for (int i = 1; i < XSize(); i++) {
        CHECK_EQ(X(0).ndim(), X(i).ndim())
            << "\nAll inputs should have the same ndim.";
        for (int j = 0; j < out_shape.size(); j++) {
            if (j == axis_) continue;
            CHECK_EQ(out_shape[j], X(i).dim(j))
                << "\nAll inputs should have the same dims"
                << ", except the concat axis.";
        }
        out_shape[axis_] += X(i).dim(axis_);
    }

    cat_dim_ = out_shape[axis_];
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->Reshape(out_shape);

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
void ConcatGradientOp<Context>::RunImpl() {
    int64_t axis_dim, cat_ofs = 0;
    auto* dy = X(-1).template data<T, Context>();

    for (int i = 0; i < YSize(); i++) {
        axis_dim = X(i).dim(axis_);
        if (Y(i)->name() != "NULL") {
            auto* dx = Y(i)->template
                mutable_data<T, Context>();
            kernel::Slice(
                outer_dim_,
                inner_dim_,
                cat_dim_,
                axis_dim,
                cat_ofs,
                dy, dx, ctx()
            );
        }
        cat_ofs += axis_dim;
    }
}

template <class Context>
void ConcatGradientOp<Context>::RunOnDevice() {
    if (X(-1).name() == "NULL") return;
    DETERMINE_RUNTIME_ARGS(X(0));

    auto out_shape = X(-1).dims();
    cat_dim_ = out_shape[axis_];
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    for (int i = 0; i < YSize(); i++)
        Y(i)->ReshapeLike(X(i));

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

DEPLOY_CPU(Concat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Concat);
#endif

DEPLOY_CPU(ConcatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ConcatGradient);
#endif

OPERATOR_SCHEMA(Concat)
     /* X(0), ... */
    .NumInputs(1, INT_MAX)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ConcatGradient)
     /* X(0), ..., dY */
    .NumInputs(2, INT_MAX)
     /* dX(0), ... */
    .NumOutputs(1, INT_MAX);

REGISTER_GRADIENT(Concat, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon