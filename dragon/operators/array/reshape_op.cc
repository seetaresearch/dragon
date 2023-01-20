#include "dragon/operators/array/reshape_op.h"
#include "dragon/core/workspace.h"

namespace dragon {

template <class Context>
void ReshapeOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  Output("X_spec")->ReshapeLike(X);

  int num_dims;
  dims(0, &num_dims);

  vec64_t in_shape(X.dims()), out_shape(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    out_shape[i] = dims(i);
  }

  int infer_axis = -1;
  int64_t total_count = 1;
  for (int i = 0; i < num_dims; ++i) {
    if (out_shape[i] == 0) {
      // Unchanged axis.
      CHECK_LT(i, (int)in_shape.size())
          << "\nUnchanged axis " << i << " is out of the "
          << "range: (0, " << in_shape.size() << ").";
      out_shape[i] = in_shape[i];
    } else if (out_shape[i] < 0) {
      // Inferred axis.
      CHECK_EQ(infer_axis, -1) << "\nCould not infer axis " << infer_axis
                               << " and " << i << " both.";
      out_shape[i] = -1, infer_axis = i;
    }
    if (out_shape[i] > 0) {
      total_count *= out_shape[i];
    }
  }

  // Determine the dimension for inferred axis.
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

  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

template <class Context>
void FlattenOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  Output("X_spec")->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), -1);
  auto out_shape = X.dims();
  auto flatten_dim = std::accumulate(
      out_shape.begin() + axis,
      out_shape.begin() + end_axis + 1,
      1,
      std::multiplies<int64_t>());
  out_shape.erase(out_shape.begin() + axis, out_shape.begin() + end_axis + 1);
  out_shape.insert(out_shape.begin() + axis, flatten_dim);
  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

template <class Context>
void SqueezeOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  Output("X_spec")->ReshapeLike(X);
  vec64_t out_shape;
  for (int i = 0; i < X.ndim(); i++) {
    if (X.dim(i) == 1) {
      bool removed = axes_.empty();
      for (auto j : axes_) {
        auto axis = j < 0 ? j + X.ndim() : j;
        CHECK(axis >= 0) << "\nExcepted the axis in [-" << X.ndim()
                         << ", INT_MAX), got " << j << ".";
        removed = (i == axis ? true : removed);
      }
      if (removed) continue;
    }
    out_shape.push_back(X.dim(i));
  }
  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

template <class Context>
void UnsqueezeOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  Output("X_spec")->ReshapeLike(X);
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
  for (size_t i = 0; i < out_shape.size(); i++) {
    out_shape[i] = out_shape[i] < 0 ? 1 : X.dim(j++);
  }
  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

DEPLOY_CPU_OPERATOR(Reshape);
DEPLOY_CPU_OPERATOR(Flatten);
DEPLOY_CPU_OPERATOR(Squeeze);
DEPLOY_CPU_OPERATOR(Unsqueeze);
DEPLOY_CPU_OPERATOR(Identity);
DEPLOY_CPU_OPERATOR(ReshapeGradient);
DEPLOY_CPU_OPERATOR(FlattenGradient);
DEPLOY_CPU_OPERATOR(SqueezeGradient);
DEPLOY_CPU_OPERATOR(UnsqueezeGradient);
REGISTER_CPU_OPERATOR(IdentityGradient, IdentityOp<CPUContext>);

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Reshape);
DEPLOY_CUDA_OPERATOR(Flatten);
DEPLOY_CUDA_OPERATOR(Squeeze);
DEPLOY_CUDA_OPERATOR(Unsqueeze);
DEPLOY_CUDA_OPERATOR(Identity);
DEPLOY_CUDA_OPERATOR(ReshapeGradient);
DEPLOY_CUDA_OPERATOR(FlattenGradient);
DEPLOY_CUDA_OPERATOR(SqueezeGradient);
DEPLOY_CUDA_OPERATOR(UnsqueezeGradient);
REGISTER_CUDA_OPERATOR(IdentityGradient, IdentityOp<CUDAContext>);
#endif

#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Reshape, Reshape);
DEPLOY_MPS_OPERATOR(Flatten, Flatten);
DEPLOY_MPS_OPERATOR(Squeeze, Squeeze);
DEPLOY_MPS_OPERATOR(Unsqueeze, Unsqueeze);
DEPLOY_MPS_OPERATOR(Identity, Identity);
DEPLOY_MPS_OPERATOR(ReshapeGradient, ReshapeGradient);
DEPLOY_MPS_OPERATOR(FlattenGradient, FlattenGradient);
DEPLOY_MPS_OPERATOR(SqueezeGradient, SqueezeGradient);
DEPLOY_MPS_OPERATOR(UnsqueezeGradient, UnsqueezeGradient);
REGISTER_MPS_OPERATOR(IdentityGradient, IdentityOp<MPSContext>);
#endif

#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Reshape);
DEPLOY_MLU_OPERATOR(Flatten);
DEPLOY_MLU_OPERATOR(Squeeze);
DEPLOY_MLU_OPERATOR(Unsqueeze);
DEPLOY_MLU_OPERATOR(Identity);
DEPLOY_MLU_OPERATOR(ReshapeGradient);
DEPLOY_MLU_OPERATOR(FlattenGradient);
DEPLOY_MLU_OPERATOR(SqueezeGradient);
DEPLOY_MLU_OPERATOR(UnsqueezeGradient);
REGISTER_MLU_OPERATOR(IdentityGradient, IdentityOp<MLUContext>);
#endif

DEFINE_OP_REPEATED_ARG(int64_t, ReshapeOp, dims);

OPERATOR_SCHEMA(Reshape).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Flatten).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Squeeze).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Unsqueeze).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Identity).AllowInplace([](int, int) -> bool { return true; });
OPERATOR_SCHEMA(ReshapeGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
OPERATOR_SCHEMA(FlattenGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
OPERATOR_SCHEMA(SqueezeGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
OPERATOR_SCHEMA(UnsqueezeGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
OPERATOR_SCHEMA(IdentityGradient).AllowInplace([](int, int) -> bool {
  return true;
});

REGISTER_GRADIENT(Reshape, SimpleGradientMaker);
REGISTER_GRADIENT(Flatten, SimpleGradientMaker);
REGISTER_GRADIENT(Squeeze, SimpleGradientMaker);
REGISTER_GRADIENT(Unsqueeze, SimpleGradientMaker);
REGISTER_GRADIENT(Identity, SimpleGradientMaker);

} // namespace dragon
