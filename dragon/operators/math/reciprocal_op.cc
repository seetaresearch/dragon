#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ReciprocalOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  math::Inv(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ReciprocalOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ReciprocalGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernel::ReciprocalGrad(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ReciprocalGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Reciprocal);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Reciprocal);
#endif

DEPLOY_CPU_OPERATOR(ReciprocalGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ReciprocalGradient);
#endif

OPERATOR_SCHEMA(Reciprocal)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(ReciprocalGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Reciprocal, InplaceGradientMaker);

} // namespace dragon
