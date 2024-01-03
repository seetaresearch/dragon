#include "dragon/operators/math/cum_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context, class Functor>
template <typename T>
void CumOp<Context, Functor>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);
  functor_.template Compute<T, Context>(
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

DEPLOY_CPU_OPERATOR(CumSum);
DEPLOY_CPU_OPERATOR(CumMax);
DEPLOY_CPU_OPERATOR(CumMin);
DEPLOY_CPU_OPERATOR(CumSumGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(CumSum);
DEPLOY_CUDA_OPERATOR(CumMax);
DEPLOY_CUDA_OPERATOR(CumMin);
DEPLOY_CUDA_OPERATOR(CumSumGradient);
#endif

OPERATOR_SCHEMA(CumSum).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(CumMax).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(CumMin).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(CumSumGradient).NumInputs(1).NumOutputs(1);

NO_GRADIENT(CumMax);
NO_GRADIENT(CumMin);
REGISTER_GRADIENT(CumSum, SimpleGradientMaker);

} // namespace dragon
