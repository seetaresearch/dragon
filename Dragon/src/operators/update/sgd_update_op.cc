#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/update/sgd_update_op.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::Compute(Tensor* dX) {
    auto* h = ws()
        ->CreateTensor("/mnt/" + slot() + "/h")
        ->ReshapeLike(*dX)
        ->template mutable_data<float, Context>();

    auto* dx = dX->template mutable_data<float, Context>();
    
    momentum_ = param("momentum");
    lr_ = param("base_lr") * lr_mult();

    // Momentum Correction, See arXiv:1706.02677
    if (last_lr_ > 0) correction_ = lr_ / last_lr_;
    last_lr_ = lr_;  // Record the last value

    kernel::SGDUpdate(
        dX->count(),
        lr_, momentum_ * correction_,
        dx, h, ctx()
    );
}

DEPLOY_CPU(SGDUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(SGDUpdate);
#endif

OPERATOR_SCHEMA(SGDUpdate)
     /* dX */
    .NumInputs(1)
     /* X */
    .NumOutputs(1);

NO_GRADIENT(SGDUpdate);

}  // namespace dragon