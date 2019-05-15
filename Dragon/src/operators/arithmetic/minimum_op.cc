#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/minimum_op.h"

namespace dragon {

template <class Context> template <typename T>
void MinimumOp<Context>::EltwiseRunImpl() {
    Y(0)->ReshapeLike(X(0));

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Minimum(
        Y(0)->count(),
        a, b,
        y, ctx()
    );
}

template <class Context> template <typename T>
void MinimumOp<Context>::BroadcastRunImpl() {
    if (X(0).count() == 1) {
        Y(0)->ReshapeLike(X(1));
        auto* a = X(0).template data<T, CPUContext>();
        auto* b = X(1).template data<T, Context>();
        auto* y = Y(0)->template mutable_data<T, Context>();
        kernel::BroadcastMinimum(
            Y(0)->count(),
            b, a[0],
            y, ctx()
        );
    } else if (X(1).count() == 1) {
        Y(0)->ReshapeLike(X(0));
        auto* a = X(0).template data<T, Context>();
        auto* b = X(1).template data<T, CPUContext>();
        auto* y = Y(0)->template mutable_data<T, Context>();
        kernel::BroadcastMinimum(
            Y(0)->count(),
            a, b[0],
            y, ctx()
        );
    } else {
        LOG(FATAL) << "Either X(0) or X(1) should be a scalar.";
    }
}

template <class Context> template <typename T>
void MinimumOp<Context>::RunImpl() {
    if (X(0).dims() == X(1).dims()) {
        EltwiseRunImpl<T>();
    } else {
        BroadcastRunImpl<T>();
    }
}

template <class Context>
void MinimumOp<Context>::RunOnDevice() {
    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void MinimumGradientOp<Context>::EltwiseRunImpl() {
    auto* a  = X(0).template data<T, Context>();
    auto* b  = X(1).template data<T, Context>();
    auto* dy = X(2).template data<T, Context>();
    auto* da = Y(0)->template mutable_data<T, Context>();
    auto* db = Y(1)->template mutable_data<T, Context>();

    kernel::MinimumGrad(
        Y(0)->count(),
        a, b, dy,
        da, db, ctx()
    );
}

template <class Context> template <typename T>
void MinimumGradientOp<Context>::BroadcastRunImpl() {
    auto* dy = X(-1).template data<T, Context>();
    if (X(0).count() == 1) {
        if (Y(0)->name() != "NULL") {
            auto* da = Y(0)->template mutable_data<T, Context>();
            math::Set(1, cast::to<T>(0.f), da, ctx());
        }
        if (Y(1)->name() != "NULL") {
            auto* a  = X(0).template data<T, CPUContext>();
            auto* b  = X(1).template data<T, Context>();
            auto* db = Y(1)->template mutable_data<T, Context>();
            kernel::BroadcastMinimumGrad(
                Y(1)->count(),
                b, a[0], dy,
                db, (T*)nullptr, ctx()
            );
        }
    } else if (X(1).count() == 1) {
        if (Y(0)->name() != "NULL") {
            auto* a  = X(0).template data<T, Context>();
            auto* b  = X(1).template data<T, CPUContext>();
            auto* da = Y(0)->template mutable_data<T, Context>();
            kernel::BroadcastMinimumGrad(
                Y(0)->count(),
                a, b[0], dy,
                da, (T*)nullptr, ctx()
            );
        }
        if (Y(1)->name() != "NULL") {
            auto* db = Y(1)->template mutable_data<T, Context>();
            math::Set(1, cast::to<T>(0.f), db, ctx());
        }
    } else {
        LOG(FATAL) << "Either X(0) or X(1) should be a scalar.";
    }
}

template <class Context> template <typename T>
void MinimumGradientOp<Context>::RunImpl() {
    if (X(0).dims() == X(1).dims()) {
        EltwiseRunImpl<T>();
    } else {
        BroadcastRunImpl<T>();
    }
}

template <class Context>
void MinimumGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));
    Y(1)->ReshapeLike(X(1));

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Minimum);
#ifdef WITH_CUDA
DEPLOY_CUDA(Minimum);
#endif

DEPLOY_CPU(MinimumGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MinimumGradient);
#endif

OPERATOR_SCHEMA(Minimum)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(MinimumGradient)
     /* A, B, dY */
    .NumInputs(3)
     /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(Minimum, SimpleGradientMaker);

}  // namespace dragon