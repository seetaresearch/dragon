#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void RMSpropUpdateOp<Context>::ComputeUpdate(Tensor* dX) {
  kernels::RMSPropUpdate(
      dX->count(),
      lr_,
      momentum_,
      decay_,
      eps_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      Slot("v")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(RMSpropUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RMSpropUpdate);
#endif

OPERATOR_SCHEMA(RMSpropUpdate).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(RMSpropUpdate);

} // namespace dragon
