#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SqrtGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  math::Div(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
  math::Scale(
      Y.count(),
      0.5f,
      dX->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SqrtGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(SqrtGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(SqrtGradient);
#endif

OPERATOR_SCHEMA(SqrtGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .Inplace({{1, 0}});

REGISTER_GRADIENT(Sqrt, InplaceGradientMaker);

} // namespace dragon
