#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void AdamUpdateOp<Context>::ComputeUpdate(Tensor* dX, Tensor* /* X */) {
  kernels::AdamUpdate(
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
void AdamWUpdateOp<Context>::ComputeUpdate(Tensor* dX, Tensor* X) {
  AdamUpdateOp<Context>::ComputeUpdate(dX, X);
  if (lambda_ > 0.f) {
    math::Axpy(
        X->count(),
        this->lr_ * lambda_,
        X->template data<float, Context>(),
        dX->template mutable_data<float, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(AdamUpdate);
DEPLOY_CPU_OPERATOR(AdamWUpdate);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(AdamUpdate);
DEPLOY_CUDA_OPERATOR(AdamWUpdate);
#endif

OPERATOR_SCHEMA(AdamUpdate).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(AdamWUpdate).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(AdamUpdate);
NO_GRADIENT(AdamWUpdate);

} // namespace dragon
