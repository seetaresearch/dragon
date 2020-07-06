#include "dragon/operators/activation/relu_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ReluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (max_value_ > 0.f) {
    kernel::ReluN(
        X.count(),
        max_value_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    kernel::Relu(
        X.count(),
        alpha_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void ReluOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ReluGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  if (max_value_ > 0.f) {
    kernel::ReluNGrad(
        Y.count(),
        max_value_,
        dY.template data<T, Context>(),
        Y.template data<T, Context>(),
        dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
        ctx());
  } else {
    kernel::ReluGrad(
        Y.count(),
        alpha_,
        dY.template data<T, Context>(),
        Y.template data<T, Context>(),
        dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void ReluGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(Relu);
#ifdef USE_CUDA
DEPLOY_CUDA(Relu);
#endif

DEPLOY_CPU(ReluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(ReluGradient);
#endif

OPERATOR_SCHEMA(Relu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .Inplace({{0, 0}});

OPERATOR_SCHEMA(ReluGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .Inplace({{1, 0}});

REGISTER_GRADIENT(Relu, InplaceGradientMaker);

} // namespace dragon
