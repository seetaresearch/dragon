#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeUpdate(Tensor* dX) {
  t_++;
  auto beta1 = Parameter("beta1"), beta2 = Parameter("beta2");
  auto coef = sqrt(1.f - pow(beta2, t_)) / (1.f - pow(beta1, t_));

  kernel::AdamUpdate(
      dX->count(),
      Parameter("base_lr") * coef * this->lr_mult_,
      beta1,
      beta2,
      Parameter("eps"),
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      Slot("v")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
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
