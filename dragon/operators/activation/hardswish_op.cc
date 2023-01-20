#include "dragon/operators/activation/hardswish_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void HardSwishOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  kernels::HardSwish(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void HardSwishGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::HardSwishGrad(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(HardSwish);
DEPLOY_CPU_OPERATOR(HardSwishGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(HardSwish);
DEPLOY_CUDA_OPERATOR(HardSwishGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(HardSwish, HardSwish);
DEPLOY_MPS_OPERATOR(HardSwishGradient, HardSwishGradient);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(HardSwishGradient);
#endif

OPERATOR_SCHEMA(HardSwish)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(HardSwishGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(HardSwish, GenericGradientMaker);

} // namespace dragon
