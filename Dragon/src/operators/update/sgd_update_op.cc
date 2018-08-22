#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/update/sgd_update_op.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::ComputeRunWithFloat32() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/sgd/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult, momentum = Param("momentum");
    //  momentum correction, see arXiv:1706.02677
    if (old_lr > 0) { correction = lr / old_lr; } old_lr = lr;
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>(ctx());

    kernel::SGDUpdate<float, Context>(Input(0).count(),
        lr, momentum * correction, dXdata, Hdata, ctx());
}

template <class Context>
void SGDUpdateOp<Context>::ComputeRunWithFloat16() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/sgd/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult, momentum = Param("momentum");
    if (old_lr > 0) { correction = lr / old_lr; } old_lr = lr;

    auto* dX32T = ws()->CreateTensor(Input(0).name() + "/f32");
    dX32T->ReshapeLike(Input(0));

    auto* dX32 = dX32T->template mutable_data<float, Context>();
    auto* dX16 = Input(0).template mutable_data<float16, Context>();
    auto* H32 = h->template mutable_data<float, Context>(ctx());

    kernel::TypeA2B<float16, float, Context>(
        Input(0).count(), dX16, dX32, ctx());
    kernel::SGDUpdate<float, Context>(Input(0).count(),
        lr, momentum * correction, dX32, H32, ctx());
}

DEPLOY_CPU(SGDUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(SGDUpdate);
#endif
OPERATOR_SCHEMA(SGDUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(SGDUpdate);

}    // namespace dragon