#include "dragon/operators/training/rmsprop_update_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void RMSPropUpdateOp<Context>::Compute(Tensor* dX) {
  auto* m = ws()->CreateTensor("/mnt/" + slot() + "/m")
                ->ReshapeLike(*dX)
                ->template mutable_data<float, Context>();
  auto* v = ws()->CreateTensor("/mnt/" + slot() + "/v")
                ->ReshapeLike(*dX)
                ->template mutable_data<float, Context>();

  kernel::RMSPropUpdate(
      dX->count(),
      param("base_lr") * lr_mult(),
      param("momentum"),
      param("decay"),
      param("eps"),
      dX->template mutable_data<float, Context>(),
      m,
      v,
      ctx());
}

DEPLOY_CPU(RMSPropUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA(RMSPropUpdate);
#endif

OPERATOR_SCHEMA(RMSPropUpdate)
    /* dX */
    .NumInputs(1)
    /* X */
    .NumOutputs(1);

NO_GRADIENT(RMSPropUpdate);

} // namespace dragon
