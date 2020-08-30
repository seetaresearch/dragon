#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::ComputeUpdate(Tensor* dX) {
  // Momentum Correction, See arXiv:1706.02677
  auto lr = Parameter("base_lr") * this->lr_mult_;
  if (last_lr_ > 0) correction_ = lr / last_lr_;
  last_lr_ = lr; // Record the last value
  kernel::SGDUpdate(
      dX->count(),
      lr,
      Parameter("momentum") * correction_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(SGDUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SGDUpdate);
#endif

OPERATOR_SCHEMA(SGDUpdate)
    /* dX */
    .NumInputs(1)
    /* X */
    .NumOutputs(1);

NO_GRADIENT(SGDUpdate);

} // namespace dragon
