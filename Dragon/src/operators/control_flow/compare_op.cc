#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "operators/control_flow/compare_op.h"

namespace dragon {

template <class Context> template <typename T>
void CompareOp<Context>::RunImpl() {
    const T* a = nullptr, * b = nullptr;

    if (X(0).count() < X(1).count()) {
        int rows, cols;
        Y(0)->ReshapeLike(X(1));
        a = ws()
            ->template data<T, Context>
                ({ X(1).count() })[0];
        b = X(1).template data<T, Context>();
        auto* ra = X(0).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 0, ra,
                const_cast<T*>(a), ctx()
            );
        } else if (utils::IsColwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 1, ra,
                const_cast<T*>(a), ctx()
            );
        } else {
            LOG(FATAL)
                << "Could not broadcast "
                << X(0).DimString()
                << " to "
                << X(1).DimString();
        }
    } else if (X(0).count() > X(1).count()) {
        int rows, cols;
        Y(0)->ReshapeLike(X(0));
        b = ws()
            ->template data<T, Context>
                ({ X(0).count() })[0];
        a = X(0).template data<T, Context>();
        auto* rb = X(1).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 0, rb,
                const_cast<T*>(b), ctx()
            );
        } else if (utils::IsColwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 1, rb,
                const_cast<T*>(b), ctx()
            );
        } else {
            LOG(FATAL)
                << "Could not broadcast "
                << X(1).DimString()
                << " to "
                << X(0).DimString();
        }
    } else {
        Y(0)->ReshapeLike(X(0));
        a = X(0).template data<T, Context>();
        b = X(1).template data<T, Context>();
    }

    auto* y = Y(0)->template mutable_data<bool, Context>();

    if (op_str_ == "EQ") {
        kernel::Equal(Y(0)->count(), a, b, y, ctx());
    } else if (op_str_ == "NE") {
        kernel::NotEqual(Y(0)->count(), a, b, y, ctx());
    } else if (op_str_ == "LT") {
        kernel::Less(Y(0)->count(), a, b, y, ctx());
    } else if (op_str_ == "GT") {
        kernel::Greater(Y(0)->count(), a, b, y, ctx());
    } else if (op_str_ == "LE") {
        kernel::LessEqual(Y(0)->count(), a, b, y, ctx());
    } else if (op_str_ == "GE") {
        kernel::GreaterEqual(Y(0)->count(), a, b, y, ctx());
    } else {
        LOG(FATAL) << "Unknown Operation: " << op_str_;
    }
}

template <class Context>
void CompareOp<Context>::RunOnDevice() {
    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));

    if (to_uint8_) {
        Y(0)->SetMeta(TypeStringToMeta("uint8"));
    }
}

DEPLOY_CPU(Compare);
#ifdef WITH_CUDA
DEPLOY_CUDA(Compare);
#endif

OPERATOR_SCHEMA(Compare)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(Compare);

}  // namespace dragon