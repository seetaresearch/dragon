#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/update/sgd_update_op.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::ComputeUpdates(Tensor* dX) {
    auto* H = ws()->CreateTensor(
        "/mnt/" + Slot() + "/sgd/h")
            ->ReshapeLike(*dX);

    lr = Param("base_lr") * this->lr_mult, momentum = Param("momentum");
    // Momentum Correction, See arXiv:1706.02677
    if (old_lr > 0) { correction = lr / old_lr; } old_lr = lr;
    auto* dXdata = dX->template mutable_data<float, Context>();
    auto* Hdata = H->template mutable_data<float, Context>();

    kernel::SGDUpdate(dX->count(), lr,
        momentum * correction, dXdata, Hdata, ctx());
}

DEPLOY_CPU(SGDUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(SGDUpdate);
#endif
OPERATOR_SCHEMA(SGDUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(SGDUpdate);

}  // namespace dragon