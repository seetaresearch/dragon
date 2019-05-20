#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "operators/control_flow/masked_assign_op.h"

namespace dragon {

template <class Context> template <typename T>
void MaskedAssignOp<Context>::RunImpl() {
    T* x = nullptr;
    auto* mask = X(1).template raw_data<Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    if (X(0).count() < Y(0)->count()) {
        int rows, cols;
        x = ws()
            ->template data<T, Context>
                ({ Y(0)->count() })[0];
        auto* rx = X(0).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                Y(0)->dims(), X(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 0,
                rx, x, ctx()
            );
        } else if (utils::IsColwiseBroadcast(
                Y(0)->dims(), X(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 1,
                rx, x, ctx()
            );
        } else {
            LOG(FATAL)
                << "Could not broadcast "
                << X(0).DimString()
                << " to "
                << Y(0)->DimString();
        }
    } else if (X(0).count() == Y(0)->count()) {
        x = const_cast<T*>(X(0)
            .template data<T, Context>());
    } else {
        LOG(FATAL)
            << "Could not assign "
            << X(0).DimString()
            << " to "
            << Y(0)->DimString();
    }

    kernel::Where(
        Y(0)->count(),
        (const uint8_t*)mask,
        x, y, y, ctx()
    );
}

template <class Context>
void MaskedAssignOp<Context>::RunOnDevice() {
    CHECK_EQ(X(1).count(), Y(0)->count())
        << "\nSize of mask and input should be equal.";

    CHECK(XIsType(X(1), bool) || XIsType(X(1), uint8_t))
        << "\nExcepted bool or uint8 mask.";

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
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