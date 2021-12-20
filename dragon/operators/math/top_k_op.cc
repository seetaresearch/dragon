#include "dragon/operators/math/top_k_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void TopKOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y_value = Output(0), *Y_index = Output(1);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  CHECK_LE(k_, C) << "\nThe top-K argument is out of the dimension.";
  auto Y_dims = X.dims();
  Y_dims[axis] = k_;

  kernels::TopK(
      X.count(0, axis),
      X.count(axis + 1),
      C,
      k_,
      largest_,
      X.template data<T, Context>(),
      Y_value->Reshape(Y_dims)->template mutable_data<T, Context>(),
      Y_index->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
      ctx());
}

template <class Context>
void TopKOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(TopK);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(TopK);
#endif

OPERATOR_SCHEMA(TopK)
    /* X */
    .NumInputs(1)
    /* Y_value, Y_index */
    .NumOutputs(2);

NO_GRADIENT(TopK);

} // namespace dragon
