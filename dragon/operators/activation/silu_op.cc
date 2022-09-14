#include "dragon/operators/activation/silu_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SiluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  kernels::Silu(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void SiluGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::SiluGrad(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Silu);
DEPLOY_CPU_OPERATOR(SiluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Silu);
DEPLOY_CUDA_OPERATOR(SiluGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Silu, Silu);
DEPLOY_MPS_OPERATOR(SiluGradient, SiluGradient);
#endif

OPERATOR_SCHEMA(Silu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SiluGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Silu, GenericGradientMaker);

} // namespace dragon
