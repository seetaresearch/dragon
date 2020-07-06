#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void NesterovUpdateOp<Context>::ComputeUpdate(Tensor* dX) {
  kernel::NesterovUpdate(
      dX->count(),
      Parameter("base_lr") * this->lr_mult_,
      Parameter("momentum"),
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

DEPLOY_CPU(NesterovUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA(NesterovUpdate);
#endif

OPERATOR_SCHEMA(NesterovUpdate)
    /* dX */
    .NumInputs(1)
    /* X */
    .NumOutputs(1);

NO_GRADIENT(NesterovUpdate);

} // namespace dragon
