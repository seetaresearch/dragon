#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void LogGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  math::Div(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void LogGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(LogGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LogGradient);
#endif

OPERATOR_SCHEMA(LogGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Log, GenericGradientMaker);

} // namespace dragon
