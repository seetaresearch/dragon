#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/update/nesterov_update_op.h"

namespace dragon {

template <class Context>
void NesterovUpdateOp<Context>::ComputeUpdates(Tensor* dX) {
    Tensor* h = ws()->CreateTensor(
        "/mnt/" + Slot() + "/nesterov/h")
            ->ReshapeLike(*dX);

    lr = Param("base_lr") * this->lr_mult, momentum = Param("momentum");
    auto* dXdata = dX->template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>();

    kernel::NesterovUpdate(dX->count(), lr,
        momentum, dXdata, Hdata, ctx());
}

DEPLOY_CPU(NesterovUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(NesterovUpdate);
#endif
OPERATOR_SCHEMA(NesterovUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(NesterovUpdate);

}  // namespace dragon