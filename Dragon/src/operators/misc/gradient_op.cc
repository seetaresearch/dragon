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
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        math::Set(Output(0)->count(),
            cast::to<T>(defaults[i]), dXdata, ctx());
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
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    int64_t count = Output(0)->count();
    if (indices.size() == 1) {
        auto* dYdata = Input(indices[0]).template data<T, Context>();
        ctx()->template Copy<T, Context, Context>(count, dXdata, dYdata);
    } else if(indices.size() == 2) {
        CHECK_EQ(count, Input(indices[1]).count());
        auto* dY1data = Input(indices[0]).template data<T, Context>();
        auto* dY2data = Input(indices[1]).template data<T, Context>();
        math::Add(count, dY1data, dY2data, dXdata, ctx());
    } else {
        size_t dy_idx = 1;
        auto* dYdata = Input(indices[0]).template data<T, Context>();
        ctx()->template Copy<T, Context, Context>(count, dXdata, dYdata);
        while (dy_idx < indices.size()) {
            if (indices.size() - dy_idx >= 2) {
                auto* dY1data = Input(indices[dy_idx]).template data<T, Context>();
                auto* dY2data = Input(indices[dy_idx + 1]).template data<T, Context>();
                kernel::GradientTwoSum(count, dY1data, dY2data, dXdata, ctx());
                dy_idx += 2;
            } else {
                dYdata = Input(indices[dy_idx]).template data<T, Context>();
                math::Add(count, dXdata, dYdata, dXdata, ctx());
                dy_idx += 1;
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
NO_GRADIENT(GradientGather);

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