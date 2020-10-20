#include "dragon/operators/array/stack_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

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

  Y->Reshape(Y_dims);
  int64_t output_offset = 0;

  for (int i = 0; i < num_stacks; i++) {
    const auto& Xi = Input(i);
    math::CopyMatrix(
        Xi.count(0, axis),
        Xi.count(axis),
        Xi.count(axis),
        Y->count(axis),
        Xi.template data<T, Context>(),
        Y->template mutable_data<T, Context>() + output_offset,
        ctx());
    output_offset += Xi.count(axis);
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

  int64_t input_offset = 0;

  for (int i = 0; i < OutputSize(); ++i) {
    auto &X = RESTORE_INPUT_SPEC(i), *dX = Output(i);
    if (dX->has_name()) {
      math::CopyMatrix(
          dY.count(0, axis),
          X.count(axis),
          dY.count(axis),
          X.count(axis),
          dY.template data<T, Context>() + input_offset,
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    }
    input_offset += X.count(axis);
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
