#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/training/update_op.h"
#include "dragon/operators/training/zero_update_op.h"

namespace dragon {

template <class Context>
template <typename T, typename CopyT>
void AdamOp<Context>::DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y) {
  kernels::Adam(
      dX->count(),
      lr_ * correction_,
      beta1_,
      beta2_,
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

template <class Context>
template <typename T, typename CopyT>
void AdamWOp<Context>::DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y) {
  kernels::AdamW(
      dX->count(),
      lr_ * correction_,
      beta1_,
      beta2_,
      eps_,
      lr_ * this->weight_decay_,
      X->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      GetState("m")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
      GetState("v")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
      X->template mutable_data<T, Context>(),
      Y ? Y->template mutable_data<CopyT, Context>() : (CopyT*)nullptr,
      ctx());
}

template <class Context>
template <typename T, typename CopyT>
void ZeroAdamWOp<Context>::DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y) {
  if (this->use_lr_scales_ > 0) {
    kernels::AdamW(
        dX->count(),
        lr_ * correction_,
        beta1_,
        beta2_,
        eps_,
        lr_ * this->weight_decay_,
        (T*)(Y->template mutable_data<CopyT, Context>() + X->count()),
        X->template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        GetState("m")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
        GetState("v")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
        X->template mutable_data<T, Context>(),
        Y->template mutable_data<CopyT, Context>(),
        ctx());
  } else {
    kernels::AdamW(
        dX->count(),
        lr_ * correction_,
        beta1_,
        beta2_,
        eps_,
        lr_ * this->weight_decay_,
        X->template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        GetState("m")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
        GetState("v")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
        X->template mutable_data<T, Context>(),
        Y->template mutable_data<CopyT, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Adam);
DEPLOY_CPU_OPERATOR(AdamW);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Adam);
DEPLOY_CUDA_OPERATOR(AdamW);
DEPLOY_CUDA_OPERATOR(ZeroAdamW);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Adam, Adam);
DEPLOY_MPS_OPERATOR(AdamW, AdamW);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Adam);
DEPLOY_MLU_OPERATOR(AdamW);
DEPLOY_MLU_OPERATOR(ZeroAdamW);
#endif

OPERATOR_SCHEMA(Adam).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(AdamW).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(ZeroAdamW).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(Adam);
NO_GRADIENT(AdamW);
NO_GRADIENT(ZeroAdamW);

} // namespace dragon
