#include "operators/update/rmsprop_update_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void RMSPropUpdateOp<Context>::ComputeRunWithFloat() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/rmsprop/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult;
    decay = Param("decay"), eps = Param("eps");
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>();

    kernel::RMSPropUpdate<float, Context>(
        Input(0).count(), lr, decay, eps, dXdata, Hdata);
}

template <class Context>
void RMSPropUpdateOp<Context>::ComputeRunWithFloat16() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/rmsprop/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult;
    decay = Param("decay"), eps = Param("eps");
    auto* dXdata = Input(0).template mutable_data<float16, Context>();
    auto* Hdata = h->template mutable_data<float16, Context>();

    kernel::RMSPropUpdate<float16, Context>(
        Input(0).count(), lr, decay, eps, dXdata, Hdata);
}

DEPLOY_CPU(RMSPropUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMSPropUpdate);
#endif
OPERATOR_SCHEMA(RMSPropUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(RMSPropUpdate);

}    // namespace dragon