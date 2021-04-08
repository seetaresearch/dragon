#include "dragon/core/workspace.h"
#include "dragon/operators/array/reshape_ops.h"

namespace dragon {

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), -1);
  SET_INPUT_SPEC(0);

  auto out_shape = X.dims();
  auto flatten_dim = std::accumulate(
      out_shape.begin() + axis,
      out_shape.begin() + end_axis + 1,
      1,
      std::multiplies<int64_t>());
  out_shape.erase(out_shape.begin() + axis, out_shape.begin() + end_axis + 1);
  out_shape.insert(out_shape.begin() + axis, flatten_dim);

  // Maybe copy the contents
  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

DEPLOY_CPU_OPERATOR(Flatten);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Flatten);
#endif

DEPLOY_CPU_OPERATOR(FlattenGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(FlattenGradient);
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
