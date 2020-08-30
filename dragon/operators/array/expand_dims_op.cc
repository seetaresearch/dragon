#include "dragon/core/workspace.h"
#include "dragon/operators/array/reshape_ops.h"

namespace dragon {

template <class Context>
void ExpandDimsOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});

  auto out_shape = vec64_t(X.ndim() + axes_.size());
  auto out_rank = (int64_t)out_shape.size();

  for (auto i : axes_) {
    CHECK(i >= -out_rank && i < out_rank)
        << "\nExcepted the axis in [-" << out_rank << ", " << out_rank
        << "), got " << i << ".";
    auto canonical_axis = i < 0 ? i + out_rank : i;
    out_shape[canonical_axis] = -1;
  }

  int64_t j = 0;
  for (size_t i = 0; i < out_shape.size(); i++)
    out_shape[i] = out_shape[i] < 0 ? 1 : X.dim(j++);

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);

  // Maybe copy the contents
  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

DEPLOY_CPU_OPERATOR(ExpandDims);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ExpandDims);
#endif

DEPLOY_CPU_OPERATOR(ExpandDimsGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ExpandDimsGradient);
#endif

OPERATOR_SCHEMA(ExpandDims)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(ExpandDimsGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(ExpandDims, SimpleGradientMaker);

} // namespace dragon
