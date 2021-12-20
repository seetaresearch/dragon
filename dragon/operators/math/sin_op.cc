#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SinGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::SinGrad(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SinGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(SinGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SinGradient);
#endif

OPERATOR_SCHEMA(SinGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Sin, GenericGradientMaker);

} // namespace dragon
