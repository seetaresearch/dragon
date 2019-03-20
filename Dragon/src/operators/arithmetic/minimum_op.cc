#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/minimum_op.h"

namespace dragon {

template <class Context> template <typename T>
void MinimumOp<Context>::EltwiseRunWithType() {
    Output(0)->ReshapeLike(Input(0));

    auto* Adata = Input(0).template data<T, Context>();
    auto* Bdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Minimum(Output(0)->count(), Adata, Bdata, Ydata, ctx());
}

template <class Context> template <typename T>
void MinimumOp<Context>::BroadcastRunWithType() {
    if (Input(0).count() == 1) {
        Output(0)->ReshapeLike(Input(1));
        auto* Adata = Input(0).template data<T, CPUContext>();
        auto* Bdata = Input(1).template data<T, Context>();
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        kernel::BroadcastMinimum(Output(0)->count(),
            Bdata, Adata[0], Ydata, ctx());
    } else if (Input(1).count() == 1) {
        Output(0)->ReshapeLike(Input(0));
        auto* Adata = Input(0).template data<T, Context>();
        auto* Bdata = Input(1).template data<T, CPUContext>();
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        kernel::BroadcastMinimum(Output(0)->count(),
            Adata, Bdata[0], Ydata, ctx());
    } else { 
        LOG(FATAL) << "Either Input(0) or Input(1) should be a scalar."; 
    }
}

template <class Context> template <typename T>
void MinimumOp<Context>::RunWithType() {
    if (Input(0).dims() == Input(1).dims()) {
        EltwiseRunWithType<T>();
    } else {
        BroadcastRunWithType<T>();
    }
}

template <class Context>
void MinimumOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(Minimum);
#ifdef WITH_CUDA
DEPLOY_CUDA(Minimum);
#endif
OPERATOR_SCHEMA(Minimum).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void MinimumGradientOp<Context>::EltwiseRunWithType() {
    auto* Adata = Input(0).template data<T, Context>();
    auto* Bdata = Input(1).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dAdata = Output(0)->template mutable_data<T, Context>();
    auto* dBdata = Output(1)->template mutable_data<T, Context>();

    kernel::MinimumGrad(Output(0)->count(),
        Adata, Bdata, dYdata, dAdata, dBdata, ctx());
}

template <class Context> template <typename T>
void MinimumGradientOp<Context>::BroadcastRunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    if (Input(0).count() == 1) {
        if (Output(0)->name() != "NULL") {
            auto* dAdata = Output(0)->template mutable_data<T, Context>();
            math::Set<T, Context>(1, cast::to<T>(0.f), dAdata, ctx());
        }
        if (Output(1)->name() != "NULL") {
            auto* Adata = Input(0).template data<T, CPUContext>();
            auto* Bdata = Input(1).template data<T, Context>();
            auto* dBdata = Output(1)->template mutable_data<T, Context>();
            kernel::BroadcastMinimumGrad(Output(1)->count(),
                Bdata, Adata[0], dYdata, dBdata, (T*)nullptr, ctx());
        }
    } else if (Input(1).count() == 1) {
        if (Output(0)->name() != "NULL") {
            auto* Adata = Input(0).template data<T, Context>();
            auto* Bdata = Input(1).template data<T, CPUContext>();
            auto* dAdata = Output(0)->template mutable_data<T, Context>();
            kernel::BroadcastMinimumGrad(Output(0)->count(),
                Adata, Bdata[0], dYdata, dAdata, (T*)nullptr, ctx());
        }
        if (Output(1)->name() != "NULL") {
            auto* dBdata = Output(1)->template mutable_data<T, Context>();
            math::Set<T, Context>(1, cast::to<T>(0.f), dBdata, ctx());
        }
    } else {
        LOG(FATAL) << "Either Input(0) or Input(1) should be a scalar."; 
    }
}

template <class Context> template <typename T>
void MinimumGradientOp<Context>::RunWithType() {
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (Input(0).dims() == Input(1).dims()) {
        EltwiseRunWithType<T>();
    } else {
        BroadcastRunWithType<T>();
    }
}

template <class Context>
void MinimumGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(MinimumGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MinimumGradient);
#endif

OPERATOR_SCHEMA(MinimumGradient)
    .NumInputs(3).NumOutputs(2);

REGISTER_GRADIENT(Minimum, SimpleGradientMaker);

}  // namespace dragon