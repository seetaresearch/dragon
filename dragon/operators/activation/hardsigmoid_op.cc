#include "dragon/operators/activation/hardsigmoid_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void HardSigmoidOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  kernels::HardSigmoid(
      X.count(),
      alpha_,
      beta_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void HardSigmoidGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::HardSigmoidGrad(
      Y.count(),
      alpha_,
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(HardSigmoid);
DEPLOY_CPU_OPERATOR(HardSigmoidGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(HardSigmoid);
DEPLOY_CUDA_OPERATOR(HardSigmoidGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(HardSigmoid, HardSigmoid);
DEPLOY_MPS_OPERATOR(HardSigmoidGradient, HardSigmoidGradient);
#endif

OPERATOR_SCHEMA(HardSigmoid)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(HardSigmoidGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(HardSigmoid, InplaceGradientMaker);

} // namespace dragon
