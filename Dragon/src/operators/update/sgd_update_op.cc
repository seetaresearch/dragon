#include "operators/update/sgd_update_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::ComputeRunWithFloat() {
    h = ws()->CreateTensor("/mnt/" + Slot() + "/sgd/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult;
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>();
    math::Axpby<float, Context>(h->count(), lr, dXdata, momentum, Hdata);
    ctx().template Copy<float, Context, Context>(h->count(), dXdata, Hdata);
}

DEPLOY_CPU(SGDUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(SGDUpdate);
#endif
OPERATOR_SCHEMA(SGDUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(SGDUpdate);

}    // namespace dragon