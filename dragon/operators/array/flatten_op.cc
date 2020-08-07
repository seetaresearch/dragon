#include "dragon/core/workspace.h"
#include "dragon/operators/array/reshape_ops.h"

namespace dragon {

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});

  vec64_t out_shape;
  if (keep_axes_ != INT_MAX) {
    if (X.ndim() < keep_axes_) {
      out_shape = vec64_t(keep_axes_, 1);
      for (int i = 0; i < X.ndim(); i++)
        out_shape[i] = X.dim(i);
    } else {
      int i = 0;
      for (; i < keep_axes_ - 1; i++)
        out_shape.push_back(X.dim(i));
      out_shape.push_back(X.count(i));
    }
  } else {
    CANONICALIZE_AXIS_WITH_TENSOR(X);
    for (int i = 0; i < axis; i++)
      out_shape.push_back(X.dim(i));
    if (num_axes_ < 1) {
      out_shape.push_back(X.count(axis));
    } else {
      auto to = axis + num_axes_;
      out_shape.push_back(X.count(axis, to));
      for (int i = to; i < X.ndim(); i++)
        out_shape.push_back(X.dim(i));
    }
  }

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);

  // Maybe copy the contents
  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

DEPLOY_CPU(Flatten);
#ifdef USE_CUDA
DEPLOY_CUDA(Flatten);
#endif

DEPLOY_CPU(FlattenGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(FlattenGradient);
#endif

OPERATOR_SCHEMA(Flatten)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(FlattenGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Flatten, SimpleGradientMaker);

} // namespace dragon
