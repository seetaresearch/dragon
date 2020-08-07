#include "dragon/operators/activation/elu_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void EluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  kernel::Elu(
      X.count(),
      alpha_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void EluOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void EluGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernel::EluGrad(
      Y.count(),
      alpha_,
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void EluGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(Elu);
#ifdef USE_CUDA
DEPLOY_CUDA(Elu);
#endif

DEPLOY_CPU(EluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(EluGradient);
#endif

OPERATOR_SCHEMA(Elu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(EluGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Elu, InplaceGradientMaker);

} // namespace dragon
