#include "dragon/operators/array/stack_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void StackOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR_AND_OFFSET(X, 1);

  int num_stacks = InputSize();

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  vec64_t Y_dims(X.dims());
  Y_dims.insert(Y_dims.begin() + axis, num_stacks);
  for (int i = 1; i < num_stacks; ++i) {
    CHECK_EQ(X.ndim(), Input(i).ndim())
        << "\nAll inputs should have the same number of dimensions.";
    for (int j = 0; j < X.ndim(); ++j) {
      CHECK_EQ(X.dim(j), Input(i).dim(j))
          << "\nAll inputs should have the same dimensions.";
    }
    STORE_INPUT_SPEC(i);
  }

  auto* y = Y->Reshape(Y_dims)->template mutable_data<T, Context>();

  for (int i = 0; i < num_stacks; i++) {
    kernel::Concat(
        X.count(0, axis),
        X.count(axis),
        1,
        num_stacks,
        i,
        Input(i).template data<T, Context>(),
        y,
        ctx());
  }
}

template <class Context>
void StackOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void StackGradientOp<Context>::DoRunWithType() {
  auto &X_ref = RESTORE_INPUT_SPEC(0), &dY = Input(0);
  CANONICALIZE_AXIS_WITH_TENSOR_AND_OFFSET(X_ref, 1)

  int num_stacks = OutputSize();
  for (int i = 0; i < num_stacks; ++i) {
    auto &X = RESTORE_INPUT_SPEC(i), *dX = Output(i);
    if (dX->has_name()) {
      kernel::Split(
          X.count(0, axis),
          X.count(axis),
          num_stacks,
          1,
          i,
          dY.template data<T, Context>(),
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void StackGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Stack);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Stack);
#endif

DEPLOY_CPU_OPERATOR(StackGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(StackGradient);
#endif

OPERATOR_SCHEMA(Stack)
    /* X(0), ... */
    .NumInputs(1, INT_MAX)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(StackGradient)
    /* dY */
    .NumInputs(1)
    /* dX(0), ... */
    .NumOutputs(1, INT_MAX);

REGISTER_GRADIENT(Stack, SimpleGradientMaker);

} // namespace dragon
