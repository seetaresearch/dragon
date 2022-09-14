#include "dragon/operators/math/arg_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ArgMaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  auto Y_dims = X.dims();
  if (!keep_dims_) {
    Y_dims.erase(Y_dims.begin() + axis);
  } else {
    Y_dims[axis] = 1;
  }

  kernels::ArgMax(
      X.count(0, axis),
      X.count(axis + 1),
      X.dim(axis),
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ArgMinOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  auto Y_dims = X.dims();
  if (!keep_dims_) {
    Y_dims.erase(Y_dims.begin() + axis);
  } else {
    Y_dims[axis] = 1;
  }

  kernels::ArgMin(
      X.count(0, axis),
      X.count(axis + 1),
      X.dim(axis),
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(ArgMax);
DEPLOY_CPU_OPERATOR(ArgMin);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ArgMax);
DEPLOY_CUDA_OPERATOR(ArgMin);
#endif

OPERATOR_SCHEMA(ArgMax).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ArgMin).NumInputs(1).NumOutputs(1);

NO_GRADIENT(ArgMax);
NO_GRADIENT(ArgMin);

} // namespace dragon
