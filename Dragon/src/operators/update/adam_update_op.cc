#include "operators/update/adam_update_op.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeRunWithFloat() {
    if (!m.get()) {
        m.reset(new Tensor()); m->ReshapeLike(Input(0));
        v.reset(new Tensor()); v->ReshapeLike(Input(0));
    }
    t++;
    coeff = sqrt(1. - pow(beta2, t)) / (1. - pow(beta1, t));
    lr = Param("base_lr") * coeff * this->lr_mult;
    kernel::AdamUpdate<float, Context>(&Input(0), 
                                         m.get(), 
                                         v.get(), 
                                           &temp,
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