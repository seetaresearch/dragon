#include "dragon/operators/math/nonzero_op.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void NonZeroOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto* X_mask = Output("X_mask")->Reshape({X.count()});
  auto* X_index = Output("X_index")->Reshape({X.count() + 1});

  // Compute the boolean mask for input elements.
  math::NotZero(
      X.count(),
      X.template data<T, Context>(),
      (bool*)X_mask->template mutable_data<uint8_t, Context>(),
      ctx());

  // Select the index of values matching the criteria.
  // The first ``num_selected`` indices are valid.
  int num_selected;
  kernels::Flagged(
      X.count(),
      X_mask->template mutable_data<uint8_t, Context>(),
      X_index->template mutable_data<int, Context>(),
      &num_selected,
      ctx());

  // Convert the flat indices into n-dimension coordinates.
  Y->Reshape({num_selected, X.ndim()});
  kernels::UnravelIndex(
      num_selected,
      X.ndim(),
      X.dims().data(),
      X_index->template data<int, Context>(),
      Y->template mutable_data<int64_t, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(NonZero);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(NonZero);
#endif

OPERATOR_SCHEMA(NonZero).NumInputs(1).NumOutputs(1);

NO_GRADIENT(NonZero);

} // namespace dragon
