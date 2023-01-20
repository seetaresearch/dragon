#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/arg_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLArgMaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  auto Y_dims = X.dims();
  if (!keep_dims_) {
    Y_dims.erase(Y_dims.begin() + axis);
  } else {
    Y_dims[axis] = 1;
  }

  impl_.Setup<T>(X.dims(), {axis}, ctx());
  impl_.ComputeIndex<T>(
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<int, Context>(),
      ctx()->workspace()->template data<Context>(impl_.scratch_size()),
      ctx());
}

template <class Context>
template <typename T>
void CNNLArgMinOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  auto Y_dims = X.dims();
  if (!keep_dims_) {
    Y_dims.erase(Y_dims.begin() + axis);
  } else {
    Y_dims[axis] = 1;
  }

  impl_.Setup<T>(X.dims(), {axis}, ctx());
  impl_.ComputeIndex<T>(
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<int, Context>(),
      ctx()->workspace()->template data<Context>(impl_.scratch_size()),
      ctx());
}

DEPLOY_CNNL_OPERATOR(ArgMax);
DEPLOY_CNNL_OPERATOR(ArgMin);

} // namespace dragon

#endif // USE_MLU
