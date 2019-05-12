#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/update/nesterov_update_op.h"

namespace dragon {

template <class Context>
void NesterovUpdateOp<Context>::Compute(Tensor* dX) {
    auto* h = ws()
        ->CreateTensor("/mnt/" + slot() + "/h")
        ->ReshapeLike(*dX)
        ->template mutable_data<float, Context>();

    auto* dx = dX->template mutable_data<float, Context>();

    momentum_ = param("momentum");
    lr_ = param("base_lr") * lr_mult();

    kernel::NesterovUpdate(
        dX->count(),
        lr_, momentum_,
        dx, h, ctx()
    );
}

DEPLOY_CPU(NesterovUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(NesterovUpdate);
#endif

OPERATOR_SCHEMA(NesterovUpdate)
     /* dX */
    .NumInputs(1)
     /* X */
    .NumOutputs(1);

NO_GRADIENT(NesterovUpdate);

}  // namespace dragon