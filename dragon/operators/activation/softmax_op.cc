#include "dragon/operators/activation/softmax_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SoftmaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  CANONICALIZE_AXIS_WITH_TENSOR(X);
  kernel::Softmax(
      X.count(0, axis),
      X.count(axis + 1),
      X.dim(axis),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SoftmaxOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SoftmaxGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(Y);
  kernel::SoftmaxGrad(
      Y.count(0, axis),
      Y.count(axis + 1),
      Y.dim(axis),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SoftmaxGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Softmax);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Softmax);
#endif

DEPLOY_CPU_OPERATOR(SoftmaxGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SoftmaxGradient);
#endif

OPERATOR_SCHEMA(Softmax)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(SoftmaxGradient)
    /* Y, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{1, 0}});

REGISTER_GRADIENT(Softmax, InplaceGradientMaker);

} // namespace dragon
