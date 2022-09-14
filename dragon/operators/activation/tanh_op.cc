#include "dragon/operators/activation/tanh_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void TanhOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  kernels::Tanh(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void TanhGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::TanhGrad(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Tanh);
DEPLOY_CPU_OPERATOR(TanhGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Tanh);
DEPLOY_CUDA_OPERATOR(TanhGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Tanh, Tanh);
DEPLOY_MPS_OPERATOR(TanhGradient, TanhGradient);
#endif

OPERATOR_SCHEMA(Tanh)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(TanhGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Tanh, InplaceGradientMaker);

} // namespace dragon
