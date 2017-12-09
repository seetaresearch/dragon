#include "operators/update/rmsprop_update_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void RMSPropUpdateOp<Context>::ComputeRunWithFloat() {
    if (!history.get()) {
        string slot = OperatorBase::GetSingleArg<string>("slot", "");
        if (slot.empty()) history.reset(new Tensor());
        else history.reset(ws()->CreateTensor("/mnt/" + name() + "/history"));
        history->ReshapeLike(input(0));
    }
    lr = param("base_lr") * this->lr_mult;
    auto* dXdata = input(0).template mutable_data<float, Context>();
    auto* Hdata = history->template mutable_data<float, Context>();
    kernel::RMSPropUpdate<float, Context>(input(0).count(), 
                                                    dXdata, 
                                                     Hdata, 
                                                     &temp, 
                                                     decay, 
                                                       eps, 
                                                       lr);
}

DEPLOY_CPU(RMSPropUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMSPropUpdate);
#endif
OPERATOR_SCHEMA(RMSPropUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(RMSPropUpdate);
    
}    // namespace dragon