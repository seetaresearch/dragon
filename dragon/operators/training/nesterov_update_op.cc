#include "dragon/operators/training/nesterov_update_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void NesterovUpdateOp<Context>::Compute(Tensor* dX) {
  auto* m = ws()->CreateTensor("/mnt/" + slot() + "/m")
                ->ReshapeLike(*dX)
                ->template mutable_data<float, Context>();

  kernel::NesterovUpdate(
      dX->count(),
      param("base_lr") * lr_mult(),
      param("momentum"),
      dX->template mutable_data<float, Context>(),
      m,
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
