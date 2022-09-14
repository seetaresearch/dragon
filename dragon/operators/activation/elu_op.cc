#include "dragon/operators/activation/elu_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void EluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  kernels::Elu(
      X.count(),
      alpha_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void EluGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::EluGrad(
      Y.count(),
      alpha_,
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Elu);
DEPLOY_CPU_OPERATOR(EluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Elu);
DEPLOY_CUDA_OPERATOR(EluGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Elu, Elu);
DEPLOY_MPS_OPERATOR(EluGradient, EluGradient);
#endif

OPERATOR_SCHEMA(Elu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(EluGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Elu, InplaceGradientMaker);

} // namespace dragon
