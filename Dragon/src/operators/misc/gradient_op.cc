#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/misc/gradient_op.h"

namespace dragon {

template <class Context> template <typename T>
void GradientGenerateOp<Context>::RunWithType() {
    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->name() == "NULL") continue;
        Output(i)->ReshapeLike(Input(i));
        auto v = cast::to<T>(defaults[i]);
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        math::Set(Output(0)->count(), v, Ydata, ctx());
    }
}

template <class Context>
void GradientGenerateOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(GradientGenerate);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientGenerate);
#endif
OPERATOR_SCHEMA(GradientGenerate);

template <class Context> template <typename T>
void GradientGatherOp<Context>::RunWithType() {
    int64_t count = Output(0)->count();
    auto* Y = Output(0)->template mutable_data<T, Context>();
    if (indices.size() == 1) {
        auto* X = Input(indices[0]).template data<T, Context>();
        ctx()->template Copy<T, Context, Context>(count, Y, X);
    } else if(indices.size() == 2) {
        CHECK_EQ(count, Input(indices[1]).count());
        auto* X1 = Input(indices[0]).template data<T, Context>();
        auto* X2 = Input(indices[1]).template data<T, Context>();
        math::Add(count, X1, X2, Y, ctx());
    } else {
        size_t index = 1;
        auto* X = Input(indices[0]).template data<T, Context>();
        ctx()->template Copy<T, Context, Context>(count, Y, X);
        while (index < indices.size()) {
            if (indices.size() - index >= 2) {
                auto* X1 = Input(indices[index]).template data<T, Context>();
                auto* X2 = Input(indices[index + 1]).template data<T, Context>();
                kernel::GradientTwoSum(count, X1, X2, Y, ctx());
                index += 2;
            } else {
                X = Input(indices[index]).template data<T, Context>();
                math::Add(count, Y, X, Y, ctx()); break;
            }
        }
    }
}

template <class Context>
void GradientGatherOp<Context>::RunOnDevice() {
    if (indices.size() == 0) return;
    Output(0)->ReshapeLike(Input(indices[0]));

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

DEPLOY_CPU(GradientGather);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientGather);
#endif
OPERATOR_SCHEMA(GradientGather).NumOutputs(1);

template <class Context> template <typename T>
void GradientAddOp<Context>::RunWithType() {
    auto* X = Input(1).template data<T, Context>();
    auto* Y = Output(0)->template mutable_data<T, Context>();
    math::Add(Output(0)->count(), Y, X, Y, ctx());
}

template <class Context>
void GradientAddOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).name(), Output(0)->name())
        << "\nRequires X(0) == Y(0).";

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

DEPLOY_CPU(GradientAdd);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientAdd);
#endif
OPERATOR_SCHEMA(GradientAdd)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context>
void StopGradientOp<Context>::RunOnDevice() {
    if (Output(0)->name() != Input(0).name()) {
        Output(0)->ReshapeLike(Input(0));
        Output(0)->template CopyFrom<Context>(Input(0), ctx());
    }
}

DEPLOY_CPU(StopGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(StopGradient);
#endif

OPERATOR_SCHEMA(StopGradient)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });;
NO_GRADIENT(StopGradient);

}  // namespace dragon