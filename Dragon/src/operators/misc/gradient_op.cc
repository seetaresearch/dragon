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
    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void GradientGatherOp<Context>::RunImpl() {
    int64_t count = Y(0)->count();
    auto* y = Y(0)->template mutable_data<T, Context>();
    if (indices.size() == 1) {
        auto* x = X(indices[0]).template data<T, Context>();
        math::Copy(count, x, y, ctx());
    } else if(indices.size() == 2) {
        CHECK_EQ(count, X(indices[1]).count());
        auto* a = X(indices[0]).template data<T, Context>();
        auto* b = X(indices[1]).template data<T, Context>();
        math::Add(count, a, b, y, ctx());
    } else {
        size_t i = 1;
        auto* x = X(indices[0]).template data<T, Context>();
        math::Copy(count, x, y, ctx());
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

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, Xi);
}

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

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

template <class Context>
void StopGradientOp<Context>::RunOnDevice() {
    if (Y(0)->name() != X(0).name()) {
        Y(0)->ReshapeLike(X(0));
        Y(0)->CopyFrom(X(0), ctx());
    }
}

DEPLOY_CPU(GradientGenerate);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientGenerate);
#endif

DEPLOY_CPU(GradientGather);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientGather);
#endif

DEPLOY_CPU(GradientAdd);
#ifdef WITH_CUDA
DEPLOY_CUDA(GradientAdd);
#endif

DEPLOY_CPU(StopGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(StopGradient);
#endif

OPERATOR_SCHEMA(GradientGenerate)
     /* X(0), ... */
    .NumInputs(1, INT_MAX)
     /* Y(0), ... */
    .NumOutputs(1, INT_MAX);

OPERATOR_SCHEMA(GradientGather)
     /* X(0), ... */
    .NumInputs(1, INT_MAX)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GradientAdd)
     /* X(0), X(1) */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1)
     /* X(0) => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(StopGradient)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

NO_GRADIENT(StopGradient);

}  // namespace dragon