#include "operators/update/sgd_update_op.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::ComputeRunWithFloat() {
    if (!history.get()) {
        history.reset(new Tensor());
        history->ReshapeLike(input(0));
    }
    lr = param("base_lr") * this->lr_mult;
    auto* dXdata = input(0).template mutable_data<float, Context>();
    auto* Hdata = history->template mutable_data<float, Context>();
    math::Axpby<float, Context>(history->count(), lr, dXdata, momentum, Hdata);
    ctx().template Copy<float, Context, Context>(history->count(), dXdata, Hdata);
}

DEPLOY_CPU(SGDUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(SGDUpdate);
#endif
OPERATOR_SCHEMA(SGDUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(SGDUpdate);

}    // namespace dragon