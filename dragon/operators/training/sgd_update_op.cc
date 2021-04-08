#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::ComputeUpdate(Tensor* dX) {
  kernels::SGDUpdate(
      dX->count(),
      lr_,
      momentum_ * correction_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(SGDUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SGDUpdate);
#endif

OPERATOR_SCHEMA(SGDUpdate).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(SGDUpdate);

} // namespace dragon
