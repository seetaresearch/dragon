#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SignGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  math::Set(
      dY.count(),
      convert::To<T>(0.f),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SignGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(SignGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SignGradient);
#endif

OPERATOR_SCHEMA(SignGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Sign, SimpleGradientMaker);

} // namespace dragon
