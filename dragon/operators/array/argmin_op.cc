#include "dragon/operators/array/arg_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ArgMinOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  // Determine the reduce scheme
  // 1) Reduce along the specified axis
  // 2) Reduce to a scalar
  int64_t outer_dim, axis_dim, inner_dim;
  if (axis != INT_MAX) {
    axis_dim = X.dim(axis);
    outer_dim = X.count(0, axis);
    inner_dim = X.count(axis + 1);
  } else {
    axis_dim = X.count();
    outer_dim = inner_dim = 1;
  }

  // Determine the output dimensions
  auto Y_dims = X.dims();
  if (!keep_dims_) {
    if (axis != INT_MAX) {
      Y_dims.erase(Y_dims.begin() + axis);
    } else {
      Y_dims = {};
    }
  } else {
    if (axis != INT_MAX) {
      Y_dims[axis] = 1;
    } else {
      Y_dims = {1};
    }
  }

  kernel::ArgMin(
      outer_dim,
      inner_dim,
      axis_dim,
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
      ctx());
}

template <class Context>
void ArgMinOp<Context>::RunOnDevice() {
  DispatchHelper<MathTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(ArgMin);
#ifdef USE_CUDA
DEPLOY_CUDA(ArgMin);
#endif

OPERATOR_SCHEMA(ArgMin)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(ArgMin);

} // namespace dragon
