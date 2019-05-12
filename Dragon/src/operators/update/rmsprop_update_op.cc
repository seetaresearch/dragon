#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/update/rmsprop_update_op.h"

namespace dragon {

template <class Context>
void RMSPropUpdateOp<Context>::Compute(Tensor* dX) {
    auto* h = ws()
        ->CreateTensor("/mnt/" + slot() + "/h")
        ->ReshapeLike(*dX)
        ->template mutable_data<float, Context>();

    auto* dx = dX->template mutable_data<float, Context>();

    lr_ = param("base_lr") * lr_mult();
    decay_ = param("decay"), eps_ = param("eps");

    kernel::RMSPropUpdate(
        dX->count(),
        lr_, decay_, eps_,
        dx, h, ctx()
    );
}

DEPLOY_CPU(RMSPropUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMSPropUpdate);
#endif

OPERATOR_SCHEMA(RMSPropUpdate)
     /* dX */
    .NumInputs(1)
     /* X */
    .NumOutputs(1);

NO_GRADIENT(RMSPropUpdate);

}  // namespace dragon