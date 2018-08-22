#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/update/rmsprop_update_op.h"

namespace dragon {

template <class Context>
void RMSPropUpdateOp<Context>::ComputeRunWithFloat32() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/rmsprop/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult;
    decay = Param("decay"), eps = Param("eps");
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>(ctx());

    kernel::RMSPropUpdate<float, Context>(
        Input(0).count(), lr, decay, eps, dXdata, Hdata, ctx());
}

template <class Context>
void RMSPropUpdateOp<Context>::ComputeRunWithFloat16() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/rmsprop/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult;
    decay = Param("decay"), eps = Param("eps");

    auto* dX32T = ws()->CreateTensor(Input(0).name() + "/f32");
    dX32T->ReshapeLike(Input(0));

    auto* dX32 = dX32T->template mutable_data<float, Context>();
    auto* dX16 = Input(0).template mutable_data<float16, Context>();
    auto* H32 = h->template mutable_data<float, Context>(ctx());

    kernel::TypeA2B<float16, float, Context>(
        Input(0).count(), dX16, dX32, ctx());
    kernel::RMSPropUpdate<float, Context>(
        Input(0).count(), lr, decay, eps, dX32, H32, ctx());
}

DEPLOY_CPU(RMSPropUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMSPropUpdate);
#endif
OPERATOR_SCHEMA(RMSPropUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(RMSPropUpdate);

}    // namespace dragon