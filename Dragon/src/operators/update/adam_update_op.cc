#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/update/adam_update_op.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeUpdates(Tensor* dX) {
    auto* M = ws()->CreateTensor(
        "/mnt/" + Slot() + "/adam/m")
            ->ReshapeLike(*dX);
    auto* V = ws()->CreateTensor(
        "/mnt/" + Slot() + "/adam/v")
            ->ReshapeLike(*dX);

    t++;
    beta1 = Param("beta1"), beta2 = Param("beta2"), eps = Param("eps");
    float coeff = sqrt(1. - pow(beta2, t)) / (1. - pow(beta1, t));
    lr = Param("base_lr") * coeff * this->lr_mult;
    auto* dXdata = dX->template mutable_data<float, Context>();
    auto* Mdata = M->template mutable_data<float, Context>();
    auto* Vdata = V->template mutable_data<float, Context>();

    kernel::AdamUpdate(dX->count(), lr, beta1,
        beta2, eps, dXdata, Mdata, Vdata, ctx());
}

DEPLOY_CPU(AdamUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(AdamUpdate);
#endif
OPERATOR_SCHEMA(AdamUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(AdamUpdate);

}  // namespace dragon