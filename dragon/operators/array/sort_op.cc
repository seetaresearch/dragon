#include "dragon/operators/array/sort_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SortOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y_value = Output(0), *Y_index = Output(1);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  kernels::TopK(
      X.count(0, axis),
      X.count(axis + 1),
      X.dim(axis),
      X.dim(axis),
      descending_ > 0 ? 1 : 0,
      X.template data<T, Context>(),
      Y_value->ReshapeLike(X)->template mutable_data<T, Context>(),
      Y_index->ReshapeLike(X)->template mutable_data<int64_t, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Sort);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Sort);
#endif

OPERATOR_SCHEMA(Sort).NumInputs(1).NumOutputs(2);

NO_GRADIENT(Sort);

} // namespace dragon
