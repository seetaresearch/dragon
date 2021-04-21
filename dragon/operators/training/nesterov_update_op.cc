#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void NesterovUpdateOp<Context>::ComputeUpdate(Tensor* dX, Tensor* /* X */) {
  kernels::NesterovUpdate(
      dX->count(),
      lr_,
      momentum_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(NesterovUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(NesterovUpdate);
#endif

OPERATOR_SCHEMA(NesterovUpdate).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(NesterovUpdate);

} // namespace dragon
