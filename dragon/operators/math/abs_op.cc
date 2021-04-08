#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AbsGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);

  // Determine the sign of input
  math::Sign(
      X.count(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());

  // Multiply the sign and input gradient to output gradient
  math::Mul(
      X.count(),
      dY.template data<T, Context>(),
      dX->template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void AbsGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(AbsGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(AbsGradient);
#endif

OPERATOR_SCHEMA(AbsGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Abs, GenericGradientMaker);

} // namespace dragon
