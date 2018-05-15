#include "operators/update/sgd_update_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::ComputeRunWithFloat() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/sgd/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult, momentum = Param("momentum");
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>();
    kernel::SGDUpdate<float, Context>(Input(0).count(),
                          lr, momentum, dXdata, Hdata);
}

DEPLOY_CPU(SGDUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(SGDUpdate);
#endif
OPERATOR_SCHEMA(SGDUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(SGDUpdate);

}    // namespace dragon