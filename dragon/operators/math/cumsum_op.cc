#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/math/cum_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CumSumOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);
  kernels::CumSum(
      X.count(0, axis),
      X.count(axis + 1),
      X.dim(axis),
      exclusive_,
      reverse_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void CumSumGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, dY.ndim(), 0);
  kernels::CumSum(
      dY.count(0, axis),
      dY.count(axis + 1),
      dY.dim(axis),
      exclusive_,
      !reverse_,
      dY.template data<T, Context>(),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void CumSumOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

template <class Context>
void CumSumGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(CumSum);
DEPLOY_CPU_OPERATOR(CumSumGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(CumSum);
DEPLOY_CUDA_OPERATOR(CumSumGradient);
#endif

OPERATOR_SCHEMA(CumSum).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(CumSumGradient).NumInputs(1).NumOutputs(1);

REGISTER_GRADIENT(CumSum, SimpleGradientMaker);

} // namespace dragon
