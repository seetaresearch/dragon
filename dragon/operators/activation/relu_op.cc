#include "dragon/operators/activation/relu_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ReluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (max_value_ > 0.f) {
    kernels::ReluN(
        X.count(),
        max_value_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    kernels::Relu(
        X.count(),
        alpha_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void ReluGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  if (max_value_ > 0.f) {
    kernels::ReluNGrad(
        Y.count(),
        max_value_,
        dY.template data<T, Context>(),
        Y.template data<T, Context>(),
        dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
        ctx());
  } else {
    kernels::ReluGrad(
        Y.count(),
        alpha_,
        dY.template data<T, Context>(),
        Y.template data<T, Context>(),
        dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Relu);
DEPLOY_CPU_OPERATOR(ReluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Relu);
DEPLOY_CUDA_OPERATOR(ReluGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Relu, Relu);
DEPLOY_MPS_OPERATOR(ReluGradient, ReluGradient);
#endif

OPERATOR_SCHEMA(Relu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(ReluGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Relu, InplaceGradientMaker);

} // namespace dragon
