#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/stack_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 0); \
    axis = axis < 0 ? axis + X.ndim() + 1 : axis; \
    CHECK(axis >= 0 && axis <= X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() + 1 << ", " << X.ndim() \
       << "], got " << OperatorBase::Arg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void StackOp<Context>::RunWithType() {
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    for (int i = 0; i < InputSize(); i++) {
        auto* Xdata = Input(i).template data<T, Context>();
        kernel::Concat(
            outer_dim, inner_dim,
                1, InputSize(),
                    i, Xdata, Ydata, ctx());
    }
}

template <class Context>
void StackOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    stack_dims = concat_dims =  Input(0).dims();
    concat_dims.insert(concat_dims.begin() + axis, InputSize());

    for (int i = 1; i < InputSize(); i++) {
        CHECK_EQ(stack_dims.size(), Input(i).ndim())
            << "\nAll inputs should have the same ndim.";
        for (int j = 0; j < stack_dims.size(); j++)
            CHECK_EQ(stack_dims[j], Input(i).dim(j))
                << "\nAll inputs should have the same dimensions.";
    }

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis);

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

DEPLOY_CPU(Stack);
#ifdef WITH_CUDA
DEPLOY_CUDA(Stack);
#endif
OPERATOR_SCHEMA(Stack).NumInputs(1, INT_MAX).NumOutputs(1);

template <class Context> template <typename T>
void StackGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();

    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->name() != "ignore") {
            auto* dXdata = Output(i)->template mutable_data<T, Context>();
            kernel::Slice(
                outer_dim, inner_dim,
                    OutputSize(), 1,
                        i, dYdata, dXdata, ctx());
        }
    }
}

template <class Context>
void StackGradientOp<Context>::RunOnDevice() {
    if (Input(-1).name() == "ignore") return;

    DETERMINE_RUNTIME_ARGUMENTS(Input(-1));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis);

    for (int i = 0; i < OutputSize(); i++)
        Output(i)->ReshapeLike(Input(i));

    if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(StackGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(StackGradient);
#endif

OPERATOR_SCHEMA(StackGradient)
    .NumInputs(2, INT_MAX).NumOutputs(1, INT_MAX);

REGISTER_GRADIENT(Stack, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon