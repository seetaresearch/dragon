#include "dragon/operators/array/sort_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SortOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y_value = Output(0), *Y_index = Output(1);
  CANONICALIZE_AXIS_WITH_TENSOR(X);
  axis = (axis == INT_MAX ? X.ndim() - 1 : axis);

  kernel::TopSelect(
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

template <class Context>
void SortOp<Context>::RunOnDevice() {
  DispatchHelper<NumericalTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Sort);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Sort);
#endif

OPERATOR_SCHEMA(Sort)
    /* X */
    .NumInputs(1)
    /* Value, Index */
    .NumOutputs(2);

NO_GRADIENT(Sort);

} // namespace dragon
