#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/index_select_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() \
        << ", " << X.ndim() << "), got " \
        << OpArg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void IndexSelectOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* i = X(1).template mutable_data<int64_t, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::IndexSelect(
        outer_dim_,
        inner_dim_,
        axis_dim_,
        nindices_,
        i, x,
        y, ctx()
    );
}

template <class Context>
void IndexSelectOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    nindices_  = X(1).count();
    axis_dim_  = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    CHECK_GT(nindices_, 0)
        << "\nLength of indices must > 0.";

    const auto& s1 = X(0).dims().begin();
    const auto& e1 = s1 + axis_, s3 = e1 + 1;
    const auto& e3 = X(0).dims().end();
    const auto& s2 = X(1).dims().begin();
    const auto& e2 = X(1).dims().end();
    vec64_t out_shape(s1, e1);
    out_shape.insert(out_shape.end(), s2, e2);
    out_shape.insert(out_shape.end(), s3, e3);

    Y(0)->Reshape(out_shape);

    CHECK(X(1).template IsType<int64_t>())
        << "\nThe type of indices should be int64.";

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
void IndexSelectGradientOp<Context>::RunImpl() {
    auto* i  = X(1).template data<int64_t, Context>();
    auto* dy = X(2).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    math::Set(
        X(0).count(),
        cast::to<T>(0.f),
        dx, ctx()
    );

    kernel::IndexSelectGrad(
        outer_dim_,
        inner_dim_,
        axis_dim_,
        nindices_,
        i, dy,
        dx, ctx()
    );
}

template <class Context>
void IndexSelectGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    nindices_  = X(1).count();
    axis_dim_  = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->ReshapeLike(X(0));

    CHECK(X(1).template IsType<int64_t>())
        << "\nThe type of indices should be int64.";

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

DEPLOY_CPU(IndexSelect);
#ifdef WITH_CUDA
DEPLOY_CUDA(IndexSelect);
#endif

DEPLOY_CPU(IndexSelectGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(IndexSelectGradient);
#endif

OPERATOR_SCHEMA(IndexSelect)
     /* X, Index */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(IndexSelectGradient)
     /* X, Index, dY */
    .NumInputs(3)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0)} )
        );
    }
};

}  // namespace

REGISTER_GRADIENT(IndexSelect, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon