#include "dragon/operators/activation/sigmoid_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SigmoidOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  kernel::Sigmoid(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SigmoidOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SigmoidGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernel::SigmoidGrad(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SigmoidGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Sigmoid);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Sigmoid);
#endif

DEPLOY_CPU_OPERATOR(SigmoidGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SigmoidGradient);
#endif

OPERATOR_SCHEMA(Sigmoid)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(SigmoidGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Sigmoid, InplaceGradientMaker);

} // namespace dragon
