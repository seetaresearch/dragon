#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/update/adam_update_op.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeRunWithFloat32() {
    Tensor* m = ws()->CreateTensor("/mnt/" + Slot() + "/adam/m");
    Tensor* v = ws()->CreateTensor("/mnt/" + Slot() + "/adam/v");
    m->ReshapeLike(Input(0));
    v->ReshapeLike(Input(0));

    t++;
    beta1 = Param("beta1"), beta2 = Param("beta2"), eps = Param("eps");
    float coeff = sqrt(1. - pow(beta2, t)) / (1. - pow(beta1, t));
    lr = Param("base_lr") * coeff * this->lr_mult;
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Mdata = m->mutable_data<float, Context>(ctx());
    auto* Vdata = v->mutable_data<float, Context>(ctx());

    kernel::AdamUpdate<float, Context>(Input(0).count(),
        lr, beta1, beta2, eps, dXdata, Mdata, Vdata, ctx());
}

template <class Context>
void AdamUpdateOp<Context>::ComputeRunWithFloat16() {
    Tensor* m = ws()->CreateTensor("/mnt/" + Slot() + "/adam/m");
    Tensor* v = ws()->CreateTensor("/mnt/" + Slot() + "/adam/v");
    m->ReshapeLike(Input(0));
    v->ReshapeLike(Input(0));

    t++;
    beta1 = Param("beta1"), beta2 = Param("beta2"), eps = Param("eps");
    float coeff = sqrt(1. - pow(beta2, t)) / (1. - pow(beta1, t));
    lr = Param("base_lr") * coeff * this->lr_mult;

    auto* dX32T = ws()->CreateTensor(Input(0).name() + "/f32");
    dX32T->ReshapeLike(Input(0));

    auto* dX32 = dX32T->template mutable_data<float, Context>();
    auto* dX16 = Input(0).template mutable_data<float16, Context>();
    auto* M32 = m->mutable_data<float, Context>(ctx());
    auto* V32 = v->mutable_data<float, Context>(ctx());

    kernel::TypeA2B<float16, float, Context>(
        Input(0).count(), dX16, dX32, ctx());
    kernel::AdamUpdate<float, Context>(Input(0).count(),
        lr, beta1, beta2, eps, dX32, M32, V32, ctx());
}

DEPLOY_CPU(AdamUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(AdamUpdate);
#endif
OPERATOR_SCHEMA(AdamUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(AdamUpdate);

}    // namespace dragon