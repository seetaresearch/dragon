#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/update/adam_update_op.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::Compute(Tensor* dX) {
    auto* m = ws()
        ->CreateTensor("/mnt/" + slot() + "/m")
        ->ReshapeLike(*dX)
        ->template mutable_data<float, Context>();

    auto* v = ws()
        ->CreateTensor("/mnt/" + slot() + "/v")
        ->ReshapeLike(*dX)
        ->template mutable_data<float, Context>();

    auto* dx = dX->template mutable_data<float, Context>();

    beta1_ = param("beta1");
    beta2_ = param("beta2");
    eps_   = param("eps");
    float coef = sqrt(1. - pow(beta2_, ++t_))
                   / (1. - pow(beta1_, t_));
    lr_ = param("base_lr") * coef * lr_mult();

    kernel::AdamUpdate(
        dX->count(),
        lr_, beta1_,
        beta2_, eps_,
        dx, m, v, ctx()
    );
}

DEPLOY_CPU(AdamUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(AdamUpdate);
#endif

OPERATOR_SCHEMA(AdamUpdate)
     /* dX */
    .NumInputs(1)
     /* X */
    .NumOutputs(1);

NO_GRADIENT(AdamUpdate);

}  // namespace dragon