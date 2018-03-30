#include "operators/update/nesterov_update_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void NesterovUpdateOp<Context>::ComputeRunWithFloat() {
    h = ws()->CreateTensor("/mnt/" + Slot() + "/nesterov/h");
    tmp = ws()->CreateTensor("/mnt/" + Slot() + "/nesterov/tmp");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult;
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>();
    kernel::NesterovUpdate<float, Context>(Input(0).count(),
                                                     dXdata,
                                                      Hdata,
                                                        tmp,
                                                   momentum,
                                                         lr,
                                                    &ctx());
}

DEPLOY_CPU(NesterovUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(NesterovUpdate);
#endif
OPERATOR_SCHEMA(NesterovUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(NesterovUpdate);

}    // namespace dragon