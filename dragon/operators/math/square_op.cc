#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SquareGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  math::Mul(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
  math::Scale(
      X.count(),
      2.f,
      dX->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SquareGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(SquareGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SquareGradient);
#endif

OPERATOR_SCHEMA(SquareGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Square, GenericGradientMaker);

} // namespace dragon
