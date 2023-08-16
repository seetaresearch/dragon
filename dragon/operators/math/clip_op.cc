#include "dragon/operators/math/clip_op.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ClipOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  kernels::Clip(
      X.count(),
      std::max(low_, float(math::Traits<T>::Lowest())),
      std::min(high_, float(math::Traits<T>::Max())),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ClipGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::ClipGrad(
      X.count(),
      std::max(low_, float(math::Traits<T>::Lowest())),
      std::min(high_, float(math::Traits<T>::Max())),
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
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Clip);
DEPLOY_MLU_OPERATOR(ClipGradient);
#endif

OPERATOR_SCHEMA(Clip).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ClipGradient).NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Clip, GenericGradientMaker);

} // namespace dragon
