#include "dragon/operators/array/split_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SplitOp<Context>::DoRunWithType() {
  auto& X = Input(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  int num_splits = OutputSize(), num_sizes = 0;
  split(0, &num_sizes);

  vec64_t X_splits;
  if (num_sizes > 0) {
    X_splits.resize(num_splits);
    for (int i = 0; i < num_splits; ++i) {
      X_splits[i] = split(i);
    }
  } else {
    auto size = (X.dim(axis) + num_splits - 1) / num_splits;
    X_splits.resize(num_splits, size);
    X_splits.back() = X.dim(axis) - size * (num_splits - 1);
  }

  // Save for the gradient computation.
  Output("X_spec")->ReshapeLike(X)->set_meta(X.meta());
  Output("X_splits")->template CopyFrom<int64_t>(X_splits);

  int64_t copy_offset = 0, total_size = 0;
  for (int i = 0; i < num_splits; ++i) {
    total_size += X_splits[i];
    CHECK(X_splits[i] > 0 && total_size <= X.dim(axis))
        << "\nIllegal size of splits: " << Tensor::DimString(X_splits)
        << " for dimension: " << X.dim(axis);
    auto* Y = Output(i);
    if (Y->has_name()) {
      vec64_t Y_dims(X.dims());
      Y_dims[axis] = X_splits[i];
      if (keep_dims_ == 0 && X_splits[i] == 1) {
        Y_dims.erase(Y_dims.begin() + axis);
      }
      if (!copy_chunks_ && axis == 0) {
        Y->Reshape(Y_dims)
            ->set_meta(X.meta())
            ->MapFrom(&X, sizeof(T) * copy_offset)
            ->set_version(0);
      } else {
        math::CopyMatrix(
            X.count(0, axis), // M
            X_splits[i] * X.count(axis + 1), // N
            X.count(axis), // ldx
            X_splits[i] * X.count(axis + 1), // ldy
            copy_offset, // x_offset
            0, // y_offset
            X.template data<T, Context>(),
            Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
            ctx());
      }
    }
    copy_offset += X_splits[i] * X.count(axis + 1);
  }
}

template <class Context>
template <typename T>
void SplitGradientOp<Context>::DoRunWithType() {
  auto* dX = Output(0)->ReshapeLike(Input("X_spec"));
  GET_OP_AXIS_ARG(axis, dX->ndim(), 0);

  int num_splits = InputSize();
  vec64_t X_splits;
  Input("X_splits").template CopyTo<int64_t>(X_splits);
  CHECK_EQ(int(X_splits.size()), num_splits);

  for (int i = 0; i < num_splits; ++i) {
    if (Input(i).has_name()) continue;
    // Set zeros for the missing gradients.
    math::Set(
        dX->count(),
        convert::To<T>(0.f),
        dX->template mutable_data<T, Context>(),
        ctx());
    break;
  }

  int64_t copy_offset = 0;
  for (int i = 0; i < num_splits; ++i) {
    auto& dY = Input(i);
    if (dY.has_name()) {
      math::CopyMatrix(
          dX->count(0, axis), // M
          X_splits[i] * dX->count(axis + 1), // N
          X_splits[i] * dX->count(axis + 1), // ldx
          dX->count(axis), // ldy
          0, // x_offset,
          copy_offset, // y_offset
          dY.template data<T, Context>(),
          dX->template mutable_data<T, Context>(),
          ctx());
    }
    copy_offset += X_splits[i] * dX->count(axis + 1);
  }
}

DEPLOY_CPU_OPERATOR(Split);
DEPLOY_CPU_OPERATOR(SplitGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Split);
DEPLOY_CUDA_OPERATOR(SplitGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Split, Split);
DEPLOY_MPS_OPERATOR(SplitGradient, SplitGradient);
#endif

OPERATOR_SCHEMA(Split)
    /* X */
    .NumInputs(1)
    /* Y(0), ... */
    .NumOutputs(1, INT_MAX);

OPERATOR_SCHEMA(SplitGradient)
    /* dY(0), ... */
    .NumInputs(1, INT_MAX)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Split, SimpleGradientMaker);

} // namespace dragon
