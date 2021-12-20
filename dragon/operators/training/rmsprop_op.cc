#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void RMSpropOp<Context>::ComputeUpdate(Tensor* dX, Tensor* /* X */) {
  kernels::RMSprop(
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

DEPLOY_CPU_OPERATOR(RMSprop);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RMSprop);
#endif

OPERATOR_SCHEMA(RMSprop).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(RMSprop);

} // namespace dragon
