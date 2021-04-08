#include "dragon/operators/array/arg_ops.h"
#include "dragon/utils/op_kernels.h"

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
void ArgMaxOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(ArgMax);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ArgMax);
#endif

OPERATOR_SCHEMA(ArgMax)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(ArgMax);

} // namespace dragon
