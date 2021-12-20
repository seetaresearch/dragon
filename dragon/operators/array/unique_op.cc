#include "dragon/operators/array/unique_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void UniqueOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_values;
  Tensor *X_index = nullptr, *Y_counts = nullptr;
  int64_t *inverse_index = nullptr, *counts = nullptr;
  if (OutputSize() == 2) {
    if (return_inverse_) {
      X_index = Output(1)->ReshapeLike(X);
      inverse_index = X_index->template mutable_data<int64_t, Context>();
    } else if (return_counts_) {
      Y_counts = Output(1)->ReshapeLike(X);
      counts = Y_counts->template mutable_data<int64_t, Context>();
    }
  } else if (OutputSize() == 3) {
    X_index = Output(1)->ReshapeLike(X);
    Y_counts = Output(2)->ReshapeLike(X);
    inverse_index = X_index->template mutable_data<int64_t, Context>();
    counts = Y_counts->template mutable_data<int64_t, Context>();
  }

  kernels::Unique(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      inverse_index,
      counts,
      &num_values,
      ctx());

  // Shrink to match the number of values
  Y->Reshape({num_values});
  if (Y_counts) Y_counts->Reshape({num_values});
}

template <class Context>
void UniqueOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Unique);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Unique);
#endif

OPERATOR_SCHEMA(Unique)
    /* X */
    .NumInputs(1)
    /* Y, InverseIndex, Counts */
    .NumOutputs(1, 3);

NO_GRADIENT(Unique);

} // namespace dragon
