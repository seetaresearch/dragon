#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "operators/control_flow/masked_assign_op.h"

namespace dragon {

template <class Context> template <typename T>
void MaskedAssignOp<Context>::RunImpl() {
    const T* x = nullptr;
    auto* mask = X(1).template raw_data<Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    if (X(0).count() < Y(0)->count()) {
        int rows, cols;
        auto* scratch = ws()
            ->template data<T, Context>
                ({ Y(0)->count() })[0];
        auto* rx = X(0).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                Y(0)->dims(), X(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 0,
                rx, scratch, ctx()
            );
        } else if (utils::IsColwiseBroadcast(
                Y(0)->dims(), X(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 1,
                rx, scratch, ctx()
            );
        } else {
            LOG(FATAL)
                << "Could not broadcast "
                << X(0).DimString()
                << " to "
                << Y(0)->DimString();
        }
        x = scratch;
    } else if (X(0).count() == Y(0)->count()) {
        x = X(0).template data<T, Context>();
    } else {
        LOG(FATAL)
            << "Could not assign "
            << X(0).DimString()
            << " to "
            << Y(0)->DimString();
    }

    kernel::MaskedAssign(
        Y(0)->count(),
        (const uint8_t*)mask,
        x, y, ctx()
    );
}

template <class Context>
void MaskedAssignOp<Context>::RunOnDevice() {
    CHECK_EQ(X(1).count(), Y(0)->count())
        << "\nSize of mask and input should be equal.";

    CHECK(XIsType(X(1), bool) || XIsType(X(1), uint8_t))
        << "\nExcepted bool or uint8 mask.";

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

DEPLOY_CPU(MaskedAssign);
#ifdef WITH_CUDA
DEPLOY_CUDA(MaskedAssign);
#endif

OPERATOR_SCHEMA(MaskedAssign)
     /* V, M */
    .NumInputs(2)
     /* X */
    .NumOutputs(1);

NO_GRADIENT(MaskedAssign);

}  // namespace dragon