#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/update/rmsprop_update_op.h"

namespace dragon {

template <class Context>
void RMSPropUpdateOp<Context>::ComputeUpdates(Tensor* dX) {
    auto* H = ws()->CreateTensor(
        "/mnt/" + Slot() + "/rmsprop/h")
            ->ReshapeLike(*dX);

    lr = Param("base_lr") * this->lr_mult;
    decay = Param("decay"), eps = Param("eps");
    auto* dXdata = dX->template mutable_data<float, Context>();
    auto* Hdata = H->template mutable_data<float, Context>();

    kernel::RMSPropUpdate(dX->count(), lr,
        decay, eps, dXdata, Hdata, ctx());
}

DEPLOY_CPU(RMSPropUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMSPropUpdate);
#endif
OPERATOR_SCHEMA(RMSPropUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(RMSPropUpdate);

}  // namespace dragon