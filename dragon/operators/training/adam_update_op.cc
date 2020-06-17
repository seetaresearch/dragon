#include "dragon/operators/training/adam_update_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::Compute(Tensor* dX) {
  auto* m = ws()->CreateTensor("/mnt/" + slot() + "/m")
                ->ReshapeLike(*dX)
                ->template mutable_data<float, Context>();
  auto* v = ws()->CreateTensor("/mnt/" + slot() + "/v")
                ->ReshapeLike(*dX)
                ->template mutable_data<float, Context>();

  t_++;
  auto beta1 = param("beta1");
  auto beta2 = param("beta2");
  auto coef = sqrt(1.f - pow(beta2, t_)) / (1.f - pow(beta1, t_));

  kernel::AdamUpdate(
      dX->count(),
      param("base_lr") * coef * lr_mult(),
      beta1,
      beta2,
      param("eps"),
      dX->template mutable_data<float, Context>(),
      m,
      v,
      ctx());
}

DEPLOY_CPU(AdamUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA(AdamUpdate);
#endif

OPERATOR_SCHEMA(AdamUpdate)
    /* dX */
    .NumInputs(1)
    /* X */
    .NumOutputs(1);

NO_GRADIENT(AdamUpdate);

} // namespace dragon
