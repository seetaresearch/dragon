#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeUpdate(Tensor* dX) {
  kernels::AdamUpdate(
      dX->count(),
      lr_,
      beta1_,
      beta2_,
      eps_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      Slot("v")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(AdamUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(AdamUpdate);
#endif

OPERATOR_SCHEMA(AdamUpdate).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(AdamUpdate);

} // namespace dragon
