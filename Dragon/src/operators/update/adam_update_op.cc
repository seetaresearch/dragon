#include "operators/update/adam_update_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeRunWithFloat() {
    m = ws()->CreateTensor("/mnt/" + Slot() + "/adam/m");
    v = ws()->CreateTensor("/mnt/" + Slot() + "/adam/v");
    tmp = ws()->CreateTensor("/mnt/" + Slot() + "/adam/tmp");
    m->ReshapeLike(Input(0));
    v->ReshapeLike(Input(0));
    t++;
    coeff = sqrt(1. - pow(beta2, t)) / (1. - pow(beta1, t));
    lr = Param("base_lr") * coeff * this->lr_mult;
    kernel::AdamUpdate<float, Context>(&Input(0),
                                       m, v, tmp,
                                           beta1,
                                           beta2,
                                             eps,
                                             lr);
}

DEPLOY_CPU(AdamUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(AdamUpdate);
#endif
OPERATOR_SCHEMA(AdamUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(AdamUpdate);

}    // namespace dragon