#include "dragon/operators/activation/hardswish_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void HardSwishOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  kernels::HardSwish(
      X.count(),
      alpha_,
      beta_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void HardSwishOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void HardSwishGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::HardSwishGrad(
      X.count(),
      alpha_,
      beta_,
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void HardSwishGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(HardSwish);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(HardSwish);
#endif

DEPLOY_CPU_OPERATOR(HardSwishGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(HardSwishGradient);
#endif

OPERATOR_SCHEMA(HardSwish)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(HardSwishGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(HardSwish, GenericGradientMaker);

} // namespace dragon
