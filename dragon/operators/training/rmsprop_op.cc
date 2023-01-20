#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/training/update_op.h"

namespace dragon {

template <class Context>
template <typename T, typename CopyT>
void RMSpropOp<Context>::DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y) {
  kernels::RMSprop(
      dX->count(),
      lr_,
      momentum_,
      alpha_,
      eps_,
      this->weight_decay_,
      X->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      GetState("m")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
      GetState("v")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
      X->template mutable_data<T, Context>(),
      Y ? Y->template mutable_data<CopyT, Context>() : (CopyT*)nullptr,
      ctx());
}

DEPLOY_CPU_OPERATOR(RMSprop);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RMSprop);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(RMSprop, RMSprop);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(RMSprop);
#endif

OPERATOR_SCHEMA(RMSprop).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(RMSprop);

} // namespace dragon
