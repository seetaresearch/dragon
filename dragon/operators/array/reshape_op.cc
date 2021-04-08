#include "dragon/core/workspace.h"
#include "dragon/operators/array/reshape_ops.h"

namespace dragon {

template <class Context>
void ReshapeOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  SET_INPUT_SPEC(0);

  int num_dims;
  dims(0, &num_dims);

  vec64_t in_shape(X.dims()), out_shape(num_dims);
  for (int i = 0; i < num_dims; ++i)
    out_shape[i] = dims(i);

  int infer_axis = -1;
  int64_t total_count = 1;
  for (int i = 0; i < num_dims; ++i) {
    if (out_shape[i] == 0) {
      // Unchanged axis
      CHECK_LT(i, (int)in_shape.size())
          << "\nUnchanged axis " << i << " is out of the "
          << "range: (0, " << in_shape.size() << ").";
      out_shape[i] = in_shape[i];
    } else if (out_shape[i] < 0) {
      // Inferred axis
      CHECK_EQ(infer_axis, -1) << "\nCould not infer axis " << infer_axis
                               << " and " << i << " both.";
      out_shape[i] = -1, infer_axis = i;
    }
    if (out_shape[i] > 0) {
      total_count *= out_shape[i];
    }
  }

  // Determine the dimension for inferred axis
  if (infer_axis != -1) {
    CHECK_EQ(X.count() % total_count, 0)
        << "\nCan not change the total size: " << X.DimString() << " -> "
        << Tensor::DimString(out_shape);
    out_shape[infer_axis] = X.count() / total_count;
  } else {
    CHECK_EQ(total_count, X.count())
        << "\nCan not change the total size: " << X.DimString() << " -> "
        << Tensor::DimString(out_shape);
  }

  // Maybe copy the contents
  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

DEPLOY_CPU_OPERATOR(Reshape);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Reshape);
#endif

DEPLOY_CPU_OPERATOR(ReshapeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ReshapeGradient);
#endif

OPERATOR_SCHEMA(Reshape)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(ReshapeGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Reshape, SimpleGradientMaker);

} // namespace dragon
