#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void AdamOp<Context>::ComputeUpdate(Tensor* dX, Tensor* /* X */) {
  kernels::Adam(
      dX->count(),
      lr_ * correction_,
      beta1_,
      beta2_,
      eps_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      Slot("v")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

template <class Context>
void AdamWOp<Context>::ComputeUpdate(Tensor* dX, Tensor* X) {
  if (lambda_ > 0.f) {
    kernels::AdamW(
        dX->count(),
        lr_ * correction_,
        beta1_,
        beta2_,
        eps_,
        this->lr_ * lambda_,
        X->template data<float, Context>(),
        dX->template mutable_data<float, Context>(),
        Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
        Slot("v")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
        ctx());
  } else {
    kernels::Adam(
        dX->count(),
        lr_ * correction_,
        beta1_,
        beta2_,
        eps_,
        dX->template mutable_data<float, Context>(),
        Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
        Slot("v")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Adam);
DEPLOY_CPU_OPERATOR(AdamW);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Adam);
DEPLOY_CUDA_OPERATOR(AdamW);
#endif

OPERATOR_SCHEMA(Adam).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(AdamW).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(Adam);
NO_GRADIENT(AdamW);

} // namespace dragon
