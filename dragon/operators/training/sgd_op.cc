#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/training/update_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T, typename CopyT>
void MomentumSGDOp<Context>::DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y) {
  kernels::MomentumSGD(
      dX->count(),
      lr_,
      momentum_,
      this->weight_decay_,
      X->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      GetState("m")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
      X->template mutable_data<T, Context>(),
      Y ? Y->template mutable_data<CopyT, Context>() : (CopyT*)nullptr,
      ctx());
}

template <class Context>
template <typename T, typename CopyT>
void NesterovSGDOp<Context>::DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y) {
  kernels::NesterovSGD(
      dX->count(),
      lr_,
      momentum_,
      this->weight_decay_,
      X->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      GetState("m")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
      X->template mutable_data<T, Context>(),
      Y ? Y->template mutable_data<CopyT, Context>() : (CopyT*)nullptr,
      ctx());
}

template <class Context>
template <typename T, typename CopyT>
void LARSOp<Context>::DoRunWithType(Tensor* dX, Tensor* X, Tensor* Y) {
  float trust_ratio = 0.f;
  if (trust_coef_ > 0.f) {
    auto* x = X->template data<T, Context>();
    auto* dx = dX->template mutable_data<T, Context>();
    float x_norm = std::sqrt(math::Dot(X->count(), x, x, ctx()));
    float dx_norm = std::sqrt(math::Dot(dX->count(), dx, dx, ctx()));
    if (x_norm > 0.f && dx_norm > 0.f) {
      trust_ratio = trust_coef_ * (x_norm / dx_norm);
    }
  }
  if (trust_ratio > 0.f) {
    math::Scale(
        dX->count(),
        trust_ratio,
        dX->template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
  kernels::MomentumSGD(
      dX->count(),
      lr_,
      momentum_,
      this->weight_decay_,
      X->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      GetState("m")->ReshapeLike(*dX)->template mutable_data<T, Context>(),
      X->template mutable_data<T, Context>(),
      Y ? Y->template mutable_data<CopyT, Context>() : (CopyT*)nullptr,
      ctx());
}

DEPLOY_CPU_OPERATOR(MomentumSGD);
DEPLOY_CPU_OPERATOR(NesterovSGD);
DEPLOY_CPU_OPERATOR(LARS);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(MomentumSGD);
DEPLOY_CUDA_OPERATOR(NesterovSGD);
DEPLOY_CUDA_OPERATOR(LARS);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(MomentumSGD, MomentumSGD);
DEPLOY_MPS_OPERATOR(NesterovSGD, NesterovSGD);
#endif

OPERATOR_SCHEMA(MomentumSGD).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(NesterovSGD).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(LARS).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(MomentumSGD);
NO_GRADIENT(NesterovSGD);
NO_GRADIENT(LARS);

} // namespace dragon
