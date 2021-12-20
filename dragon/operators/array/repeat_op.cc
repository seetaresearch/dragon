#include "dragon/operators/array/repeat_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RepeatOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), INT_MAX);

  // Determine the repeat scheme.
  // 1) Repeat to a flatten vector if axis is not specified.
  // 2) Repeat along the specified axis.
  int64_t N, C, S;
  int64_t reps = repeats();
  if (axis == INT_MAX) {
    N = S = 1;
    C = X.count();
    Y->Reshape({C * reps});
  } else {
    C = X.dim(axis);
    N = X.count(0, axis);
    S = X.count(axis + 1);
    auto Y_dims = X.dims();
    Y_dims[axis] *= reps;
    Y->Reshape(Y_dims);
  }

  // Dispatch the repeat kenrel.
  kernels::Repeat(
      N,
      S,
      C,
      reps,
      X.template data<T, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RepeatGradientOp<Context>::DoRunWithType() {
  auto &X = Input("X_spec"), &dY = Input(0), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), INT_MAX);

  // Determine the repeat scheme
  int64_t N, C, S;
  if (axis == INT_MAX) {
    N = S = 1;
    C = X.count();
  } else {
    N = X.count(0, axis);
    C = X.dim(axis);
    S = X.count(axis + 1);
  }

  // Reduce the gradient along the axis
  kernels::RepeatGrad(
      N,
      S,
      C,
      repeats(),
      dY.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
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
