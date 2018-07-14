#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/update/nesterov_update_op.h"

namespace dragon {

template <class Context>
void NesterovUpdateOp<Context>::ComputeRunWithFloat() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/nesterov/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult, momentum = Param("momentum");
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Hdata = h->template mutable_data<float, Context>();

    kernel::NesterovUpdate<float, Context>(
        Input(0).count(), lr, momentum, dXdata, Hdata);
}

template <class Context>
void NesterovUpdateOp<Context>::ComputeRunWithFloat16() {
    Tensor* h = ws()->CreateTensor("/mnt/" + Slot() + "/nesterov/h");
    h->ReshapeLike(Input(0));

    lr = Param("base_lr") * this->lr_mult, momentum = Param("momentum");
    auto* dXdata = Input(0).template mutable_data<float16, Context>();
    auto* Hdata = h->template mutable_data<float16, Context>();

    kernel::NesterovUpdate<float16, Context>(
        Input(0).count(), lr, momentum, dXdata, Hdata);
}

DEPLOY_CPU(NesterovUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(NesterovUpdate);
#endif
OPERATOR_SCHEMA(NesterovUpdate).NumInputs(1).NumOutputs(1);

NO_GRADIENT(NesterovUpdate);

}    // namespace dragon