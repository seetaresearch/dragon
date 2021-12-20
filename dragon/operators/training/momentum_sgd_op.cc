#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
void MomentumSGDOp<Context>::ComputeUpdate(Tensor* dX, Tensor* /* X */) {
  kernels::MomentumSGD(
      dX->count(),
      lr_,
      momentum_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

template <class Context>
void NesterovSGDOp<Context>::ComputeUpdate(Tensor* dX, Tensor* /* X */) {
  kernels::NesterovSGD(
      dX->count(),
      lr_,
      momentum_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
      ctx());
}

template <class Context>
void LARSOp<Context>::ComputeUpdate(Tensor* dX, Tensor* X) {
  float trust_ratio = 0.f;
  if (trust_coef_ > 0.f) {
    auto* x = X->template data<float, Context>();
    auto* dx = dX->template mutable_data<float, Context>();
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
        dX->template data<float, Context>(),
        dX->template mutable_data<float, Context>(),
        ctx());
  }
  kernels::MomentumSGD(
      dX->count(),
      lr_,
      momentum_,
      dX->template mutable_data<float, Context>(),
      Slot("m")->ReshapeLike(*dX)->template mutable_data<float, Context>(),
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

OPERATOR_SCHEMA(MomentumSGD).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(NesterovSGD).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(LARS).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

NO_GRADIENT(MomentumSGD);
NO_GRADIENT(NesterovSGD);
NO_GRADIENT(LARS);

} // namespace dragon
