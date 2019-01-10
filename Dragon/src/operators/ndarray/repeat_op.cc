#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/ndarray/repeat_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", INT_MAX); \
    if (axis != INT_MAX) { \
        axis = axis < 0 ? axis + X.ndim() : axis; \
        CHECK(axis >= 0 && axis < X.ndim()) \
            << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
            << "), got " << OperatorBase::Arg<int64_t>("axis", 0) << "."; \
    }

template <class Context> template <typename T>
void RepeatOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Repeat(
        outer_dim, repeat_dim, inner_dim,
            repeats(), Xdata, Ydata, ctx());
}

template <class Context>
void RepeatOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    if (axis == INT_MAX) {
        outer_dim = inner_dim = 1;
        repeat_dim = Input(0).count();
        Output(0)->Reshape({ repeat_dim * repeats() });
    } else {
        outer_dim = Input(0).count(0, axis);
        repeat_dim = Input(0).dim(axis);
        inner_dim = Input(0).count(axis + 1);
        auto dims = Input(0).dims();
        dims[axis] *= repeats();
        Output(0)->Reshape(dims);
    }

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

DEPLOY_CPU(Repeat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Repeat);
#endif
OPERATOR_SCHEMA(Repeat).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void RepeatGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::RepeatGrad(
        outer_dim, repeat_dim, inner_dim,
            repeats(), dYdata, dXdata, ctx());
}

template <class Context>
void RepeatGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    if (axis == INT_MAX) {
        outer_dim = inner_dim = 1;
        repeat_dim = Input(0).count();
    } else {
        outer_dim = Input(0).count(0, axis);
        repeat_dim = Input(0).dim(axis);
        inner_dim = Input(0).count(axis + 1);
    }
    Output(0)->ReshapeLike(Input(0));

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

DEPLOY_CPU(RepeatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RepeatGradient);
#endif

OPERATOR_SCHEMA(RepeatGradient)
    .NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Repeat, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon