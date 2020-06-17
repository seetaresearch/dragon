#include "dragon/operators/array/non_zero_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void NonZeroOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto* X_mask = Buffer("X_mask")->Reshape({X.count()});
  auto* X_index = Buffer("X_index")->Reshape({X.count() + 1});

  // Compute the boolean mask for input elements
  math::NotZero(
      X.count(),
      X.template data<T, Context>(),
      (bool*)X_mask->template mutable_data<uint8_t, Context>(),
      ctx());

  // Determine the scratch requirement
  size_t scratch_size = 0;
  kernel::Flagged(
      X.count(),
      X_mask->template mutable_data<uint8_t, Context>(),
      X_index->template mutable_data<int, Context>(),
      nullptr,
      nullptr,
      scratch_size,
      ctx());

  // Select the index of values matching the criteria
  // The first ``num_selected`` indices are valid
  int num_selected;
  kernel::Flagged(
      X.count(),
      X_mask->template mutable_data<uint8_t, Context>(),
      X_index->template mutable_data<int, Context>(),
      &num_selected,
      ws()->template data<Context>({scratch_size})[0],
      scratch_size,
      ctx());

  // Convert the flat indices into n-dimension coordinates
  Y->Reshape({num_selected, X.ndim()});
  kernel::UnravelIndex(
      num_selected,
      X.ndim(),
      X.dims().data(),
      X_index->template data<int, Context>(),
      Y->template mutable_data<int64_t, Context>(),
      ctx());
}

template <class Context>
void NonZeroOp<Context>::RunOnDevice() {
  DispatchHelper<AllTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(NonZero);
#ifdef USE_CUDA
DEPLOY_CUDA(NonZero);
#endif

OPERATOR_SCHEMA(NonZero)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(NonZero);

} // namespace dragon
