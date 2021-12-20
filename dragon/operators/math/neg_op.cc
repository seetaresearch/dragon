#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void NegOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  math::Neg(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void NegOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Signed>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void NegGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  math::Neg(
      dY.count(),
      dY.template data<T, Context>(),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void NegGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Neg);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Neg);
#endif

DEPLOY_CPU_OPERATOR(NegGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(NegGradient);
#endif

OPERATOR_SCHEMA(Neg)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(NegGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Neg, SimpleGradientMaker);

} // namespace dragon
