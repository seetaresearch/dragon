#include "dragon/core/workspace.h"
#include "dragon/operators/array/cum_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void CumSumOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  kernel::CumSum(
      X.count(0, axis),
      X.dim(axis),
      X.count(axis + 1),
      exclusive_,
      reverse_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void CumSumOp<Context>::RunOnDevice() {
  DispatchHelper<MathTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CumSumGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(dY);

  kernel::CumSum(
      dY.count(0, axis),
      dY.dim(axis),
      dY.count(axis + 1),
      exclusive_,
      !reverse_,
      dY.template data<T, Context>(),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void CumSumGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(CumSum);
#ifdef USE_CUDA
DEPLOY_CUDA(CumSum);
#endif

DEPLOY_CPU(CumSumGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(CumSumGradient);
#endif

OPERATOR_SCHEMA(CumSum)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(CumSumGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(CumSum, SimpleGradientMaker);

} // namespace dragon
