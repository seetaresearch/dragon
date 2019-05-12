#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/maximum_op.h"

namespace dragon {

template <class Context> template <typename T>
void MaximumOp<Context>::EltwiseRunImpl() {
    Y(0)->ReshapeLike(X(0));

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Maximum(
        Y(0)->count(),
        a, b,
        y, ctx()
    );
}

template <class Context> template <typename T>
void MaximumOp<Context>::BroadcastRunImpl() {
    if (X(0).count() == 1) {
        Y(0)->ReshapeLike(X(1));
        auto* a = X(0).template data<T, CPUContext>();
        auto* b = X(1).template data<T, Context>();
        auto* y = Y(0)->template mutable_data<T, Context>();
        kernel::BroadcastMaximum(
            Y(0)->count(),
            b, a[0],
            y, ctx()
        );
    } else if (X(1).count() == 1) {
        Y(0)->ReshapeLike(X(0));
        auto* a = X(0).template data<T, Context>();
        auto* b = X(1).template data<T, CPUContext>();
        auto* y = Y(0)->template mutable_data<T, Context>();
        kernel::BroadcastMaximum(
            Y(0)->count(),
            a, b[0],
            y, ctx()
        );
    } else { 
        LOG(FATAL) << "Either X(0) or X(1) should be a scalar.";
    }
}

template <class Context> template <typename T>
void MaximumOp<Context>::RunImpl() {
    if (X(0).dims() == X(1).dims()) {
        EltwiseRunImpl<T>();
    } else {
        BroadcastRunImpl<T>();
    }
}

template <class Context>
void MaximumOp<Context>::RunOnDevice() {
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

template <class Context> template <typename T>
void MaximumGradientOp<Context>::EltwiseRunImpl() {
    auto* a  = X(0).template data<T, Context>();
    auto* b  = X(1).template data<T, Context>();
    auto* dy = X(2).template data<T, Context>();
    auto* da = Y(0)->template mutable_data<T, Context>();
    auto* db = Y(1)->template mutable_data<T, Context>();

    kernel::MaximumGrad(
        Y(0)->count(),
        a, b, dy,
        da, db, ctx()
    );
}

template <class Context> template <typename T>
void MaximumGradientOp<Context>::BroadcastRunImpl() {
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
            kernel::BroadcastMaximumGrad(
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
            kernel::BroadcastMaximumGrad(
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
void MaximumGradientOp<Context>::RunImpl() {
    Y(0)->ReshapeLike(X(0));
    Y(1)->ReshapeLike(X(1));

    if (X(0).dims() == X(1).dims()) {
        EltwiseRunImpl<T>();
    } else {
        BroadcastRunImpl<T>();
    }
}

template <class Context>
void MaximumGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(Maximum);
#ifdef WITH_CUDA
DEPLOY_CUDA(Maximum);
#endif

DEPLOY_CPU(MaximumGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MaximumGradient);
#endif

OPERATOR_SCHEMA(Maximum)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(MaximumGradient)
     /* A, B, dY */
    .NumInputs(3)
     /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(Maximum, SimpleGradientMaker);

}  // namespace dragon