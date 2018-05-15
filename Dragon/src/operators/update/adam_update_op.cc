#include "operators/update/adam_update_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeRunWithFloat() {
    Tensor* m = ws()->CreateTensor("/mnt/" + Slot() + "/adam/m");
    Tensor* v = ws()->CreateTensor("/mnt/" + Slot() + "/adam/v");
    m->ReshapeLike(Input(0));
    v->ReshapeLike(Input(0));

    t++;
    beta1 = Param("beta1"), beta2 = Param("beta2"), eps = Param("eps");
    float coeff = sqrt(1. - pow(beta2, t)) / (1. - pow(beta1, t));
    lr = Param("base_lr") * coeff * this->lr_mult;
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Mdata = m->mutable_data<float, Context>();
    auto* Vdata = v->mutable_data<float, Context>();
    kernel::AdamUpdate<float, Context>(Input(0).count(),
           lr, beta1, beta2, eps, dXdata, Mdata, Vdata);
}

DEPLOY_CPU(AdamUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(AdamUpdate);
#endif
OPERATOR_SCHEMA(AdamUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(AdamUpdate);

}    // namespace dragon