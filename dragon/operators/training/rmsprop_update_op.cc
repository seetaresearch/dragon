#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void RMSpropUpdateOp<Context>::ComputeUpdate(Tensor* dX) {
  kernel::RMSPropUpdate(
      dX->count(),
      Parameter("base_lr") * this->lr_mult_,
      Parameter("momentum"),
      Parameter("decay"),
      Parameter("eps"),
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      Slot("v")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

DEPLOY_CPU(RMSpropUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA(RMSpropUpdate);
#endif

OPERATOR_SCHEMA(RMSpropUpdate)
    /* dX */
    .NumInputs(1)
    /* X */
    .NumOutputs(1);

NO_GRADIENT(RMSpropUpdate);

} // namespace dragon
