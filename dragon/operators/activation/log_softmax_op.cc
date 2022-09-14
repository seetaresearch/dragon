#include "dragon/operators/activation/log_softmax_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void LogSoftmaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  kernels::LogSoftmax(
      X.count(0, axis),
      X.count(axis + 1),
      X.dim(axis),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void LogSoftmaxGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, Y.ndim(), -1);
  kernels::LogSoftmaxGrad(
      Y.count(0, axis),
      Y.count(axis + 1),
      Y.dim(axis),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(LogSoftmax);
DEPLOY_CPU_OPERATOR(LogSoftmaxGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LogSoftmax);
DEPLOY_CUDA_OPERATOR(LogSoftmaxGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(LogSoftmax, LogSoftmax);
DEPLOY_MPS_OPERATOR(LogSoftmaxGradient, LogSoftmaxGradient);
#endif

OPERATOR_SCHEMA(LogSoftmax)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(LogSoftmaxGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(LogSoftmax, InplaceGradientMaker);

} // namespace dragon
