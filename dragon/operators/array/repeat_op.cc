#include "dragon/operators/array/repeat_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RepeatOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  // Determine the repeat scheme
  // 1) Repeat to a flatten vector if axis is not specified
  // 2) Repeat along the specified axis
  int64_t outer_dim, axis_dim, inner_dim;
  if (axis == INT_MAX) {
    outer_dim = inner_dim = 1;
    axis_dim = X.count();
    Y->Reshape({axis_dim * repeats()});
  } else {
    axis_dim = X.dim(axis);
    outer_dim = X.count(0, axis);
    inner_dim = X.count(axis + 1);
    auto Y_dims = X.dims();
    Y_dims[axis] *= repeats();
    Y->Reshape(Y_dims);
  }

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);

  // Dispatch the repeat kenrel
  kernel::Repeat(
      outer_dim,
      inner_dim,
      axis_dim,
      repeats(),
      X.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void RepeatOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void RepeatGradientOp<Context>::DoRunWithType() {
  auto &X = RESTORE_INPUT_SPEC(0), &dY = Input(0), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  // Determine the repeat scheme
  int64_t outer_dim, axis_dim, inner_dim;
  if (axis == INT_MAX) {
    outer_dim = inner_dim = 1;
    axis_dim = X.count();
  } else {
    outer_dim = X.count(0, axis);
    axis_dim = X.dim(axis);
    inner_dim = X.count(axis + 1);
  }

  // Reduce the gradient along the axis
  kernel::RepeatGrad(
      outer_dim,
      inner_dim,
      axis_dim,
      repeats(),
      dY.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void RepeatGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Repeat);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Repeat);
#endif

DEPLOY_CPU_OPERATOR(RepeatGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RepeatGradient);
#endif

OPERATOR_SCHEMA(Repeat)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(RepeatGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Repeat, SimpleGradientMaker);

} // namespace dragon
