#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/concat_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 0); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void ConcatOp<Context>::RunWithType() {
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    concat_offset = 0;

    for (int i = 0; i < InputSize(); i++) {
        auto* Xdata = Input(i).template data<T, Context>();
        int64_t count = Input(i).count();
        x_concat_dim = Input(i).dim(axis);

        kernel::Concat(
            outer_dim, inner_dim,
                x_concat_dim, y_concat_dim,
                    concat_offset, Xdata, Ydata, ctx());

        concat_offset += x_concat_dim;
    }
}

template <class Context>
void ConcatOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    concat_dims = Input(0).dims();
    for (int i = 1; i < InputSize(); i++) {
        CHECK_EQ((int)concat_dims.size(), Input(i).ndim())
            << "\nAll inputs should have the same ndim.";
        for (int j = 0; j < concat_dims.size(); j++) {
            if (j == axis) continue;
            CHECK_EQ(concat_dims[j], Input(i).dim(j))
                << "\nAll inputs should have the same dimensions"
                << ", except the concat axis.";
        }
        concat_dims[axis] += Input(i).dim(axis);
    }

    y_concat_dim = concat_dims[axis];
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);

    Output(0)->Reshape(concat_dims);

    if (XIsType(Input(0), bool)) RunWithType<bool>();
    else if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), { 
        "bool", "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(Concat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Concat);
#endif
OPERATOR_SCHEMA(Concat).NumInputs(1, INT_MAX).NumOutputs(1);

template <class Context> template    <typename T>
void ConcatGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();

    concat_offset = 0;

    for (int i = 0; i < OutputSize(); i++) {
        x_concat_dim = Input(i).dim(axis);
        if (Output(i)->name() != "ignore") {
            auto* dXdata = Output(i)->template mutable_data<T, Context>();
            kernel::Slice(
                outer_dim, inner_dim,
                    y_concat_dim, x_concat_dim,
                        concat_offset, dYdata, dXdata, ctx());
        }
        concat_offset += x_concat_dim;
    }
}

template <class Context>
void ConcatGradientOp<Context>::RunOnDevice() {
    if (Input(-1).name() == "ignore") return;

    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    concat_dims = Input(-1).dims();
    y_concat_dim = concat_dims[axis];
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);

    for (int i = 0; i < OutputSize(); i++)
        Output(i)->ReshapeLike(Input(i));

    if (XIsType(Input(0), bool)) RunWithType<bool>();
    else if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "bool", "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(ConcatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ConcatGradient);
#endif

OPERATOR_SCHEMA(ConcatGradient)
    .NumInputs(2, INT_MAX).NumOutputs(1, INT_MAX);

REGISTER_GRADIENT(Concat, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon