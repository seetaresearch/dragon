#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SignGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  math::Set(
      dY.count(),
      cast::to<T>(0.f),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SignGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(SignGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(SignGradient);
#endif

OPERATOR_SCHEMA(SignGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .Inplace({{0, 0}});

REGISTER_GRADIENT(Sign, SimpleGradientMaker);

} // namespace dragon
