#include "dragon/operators/math/clip_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ClipOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto limits = this->template GetLimits<T>();
  kernels::Clip(
      X.count(),
      limits.first,
      limits.second,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ClipGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  auto limits = this->template GetLimits<T>();
  kernels::ClipGrad(
      X.count(),
      limits.first,
      limits.second,
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Clip);
DEPLOY_CPU_OPERATOR(ClipGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Clip);
DEPLOY_CUDA_OPERATOR(ClipGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Clip, Clip);
DEPLOY_MPS_OPERATOR(ClipGradient, ClipGradient);
#endif

OPERATOR_SCHEMA(Clip).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ClipGradient).NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Clip, GenericGradientMaker);

} // namespace dragon
