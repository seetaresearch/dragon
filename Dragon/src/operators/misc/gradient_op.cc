#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/misc/gradient_op.h"

namespace dragon {

template <class Context> template <typename T>
void GradientGenerateOp<Context>::RunImpl() {
    for (int i = 0; i < YSize(); i++) {
        if (Y(i)->name() == "NULL") continue;
        Y(i)->ReshapeLike(X(i));
        auto v = cast::to<T>(defaults[i]);
        auto* Ydata = Y(0)->template mutable_data<T, Context>();
        math::Set(Y(0)->count(), v, Ydata, ctx());
    }
}

template <class Context>
void GradientGenerateOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(GradientGenerate);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientGenerate);
#endif
OPERATOR_SCHEMA(GradientGenerate);

template <class Context> template <typename T>
void GradientGatherOp<Context>::RunImpl() {
    int64_t count = Y(0)->count();
    auto* y = Y(0)->template mutable_data<T, Context>();
    if (indices.size() == 1) {
        auto* x = X(indices[0]).template data<T, Context>();
        ctx()->template Copy<T, Context, Context>(count, y, x);
    } else if(indices.size() == 2) {
        CHECK_EQ(count, X(indices[1]).count());
        auto* a = X(indices[0]).template data<T, Context>();
        auto* b = X(indices[1]).template data<T, Context>();
        math::Add(count, a, b, y, ctx());
    } else {
        size_t i = 1;
        auto* x = X(indices[0]).template data<T, Context>();
        ctx()->template Copy<T, Context, Context>(count, y, x);
        while (i < indices.size()) {
            if (indices.size() - i >= 2) {
                auto* a = X(indices[i]).template data<T, Context>();
                auto* b = X(indices[i + 1]).template data<T, Context>();
                kernel::GradientTwoSum(count, a, b, y, ctx()); i += 2;
            } else {
                x = X(indices[i]).template data<T, Context>();
                math::Add(count, y, x, y, ctx()); break;
            }
        }
    }
}

template <class Context>
void GradientGatherOp<Context>::RunOnDevice() {
    if (indices.size() == 0) return;
    
    auto& Xi = X(indices[0]);
    Y(0)->ReshapeLike(Xi);

    if (XIsType(Xi, int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(Xi, uint8_t)) {
        RunImpl<uint8_t>();
    } else if (XIsType(Xi, int)) {
        RunImpl<int>();
    } else if (XIsType(Xi, int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(Xi, float16)) {
        RunImpl<float16>();
    } else if (XIsType(Xi, float)) {
        RunImpl<float>();
    } else if (XIsType(Xi, double)) {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(Xi, {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(GradientGather);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientGather);
#endif
OPERATOR_SCHEMA(GradientGather).NumOutputs(1);

template <class Context> template <typename T>
void GradientAddOp<Context>::RunImpl() {
    auto* x = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Add(Y(0)->count(), y, x, y, ctx());
}

template <class Context>
void GradientAddOp<Context>::RunOnDevice() {
    CHECK_EQ(X(0).name(), Y(0)->name())
        << "\nRequires X(0) == Y(0).";

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

DEPLOY_CPU(GradientAdd);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientAdd);
#endif

OPERATOR_SCHEMA(GradientAdd)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context>
void StopGradientOp<Context>::RunOnDevice() {
    if (Y(0)->name() != X(0).name()) {
        Y(0)->ReshapeLike(X(0));
        Y(0)->CopyFrom(X(0), ctx());
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