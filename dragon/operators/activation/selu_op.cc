#include "dragon/operators/activation/selu_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SeluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  kernel::Selu(
      X.count(),
      alpha_,
      gamma_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SeluOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SeluGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernel::SeluGrad(
      Y.count(),
      alpha_,
      gamma_,
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SeluGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(Selu);
#ifdef USE_CUDA
DEPLOY_CUDA(Selu);
#endif

DEPLOY_CPU(SeluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(SeluGradient);
#endif

OPERATOR_SCHEMA(Selu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(SeluGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Selu, InplaceGradientMaker);

} // namespace dragon
