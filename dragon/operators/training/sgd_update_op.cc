#include "dragon/operators/training/sgd_update_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void SGDUpdateOp<Context>::Compute(Tensor* dX) {
  auto* m = ws()->CreateTensor("/mnt/" + slot() + "/m")
                ->ReshapeLike(*dX)
                ->template mutable_data<float, Context>();

  // Momentum Correction, See arXiv:1706.02677
  auto lr = param("base_lr") * lr_mult();
  if (last_lr_ > 0) correction_ = lr / last_lr_;
  last_lr_ = lr; // Record the last value

  kernel::SGDUpdate(
      dX->count(),
      lr,
      param("momentum") * correction_,
      dX->template mutable_data<float, Context>(),
      m,
      ctx());
}

DEPLOY_CPU(SGDUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA(SGDUpdate);
#endif

OPERATOR_SCHEMA(SGDUpdate)
    /* dX */
    .NumInputs(1)
    /* X */
    .NumOutputs(1);

NO_GRADIENT(SGDUpdate);

} // namespace dragon
