#include "dragon/operators/array/split_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(tensor)                                       \
  auto size_splits = OP_REPEATED_ARG(int64_t, "size_splits");                \
  auto slice_points = OP_REPEATED_ARG(int64_t, "slice_points");              \
  if (!slice_points.empty()) {                                               \
    int64_t index = 0;                                                       \
    size_splits = vec64_t(num_splits);                                       \
    for (int i = 0; i < num_splits; i++) {                                   \
      size_splits[i] = i < num_splits - 1 ? slice_points[i] - index          \
                                          : tensor.dim(axis) - index;        \
      index += size_splits[i];                                               \
    }                                                                        \
  } else if (size_splits.empty()) {                                          \
    auto dim = (tensor.dim(axis) + num_splits - 1) / num_splits;             \
    size_splits = vec64_t(num_splits, dim);                                  \
    size_splits[num_splits - 1] = tensor.dim(axis) - dim * (num_splits - 1); \
  }

template <class Context>
template <typename T>
void SplitOp<Context>::DoRunWithType() {
  auto& X = Input(0);

  int num_splits = OutputSize();
  CANONICALIZE_AXIS_WITH_TENSOR(X);
  DETERMINE_RUNTIME_ARGS(X);

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);

  int64_t index = 0, next_index;
  vec64_t Y_dims(X.dims());
  for (int i = 0; i < num_splits; ++i) {
    next_index = index + size_splits[i];
    CHECK(size_splits[i] > 0 && next_index <= X.dim(axis))
        << "\nIllegal size of splits: " << Tensor::DimString(size_splits)
        << " for dimension: " << X.dim(axis);
    auto* Y = Output(i);
    if (Y->has_name()) {
      Y_dims[axis] = size_splits[i];
      kernel::Split(
          X.count(0, axis),
          X.count(axis + 1),
          X.dim(axis),
          size_splits[i],
          index,
          X.template data<T, Context>(),
          Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
          ctx());
    }
    index = next_index;
  }
}

template <class Context>
void SplitOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SplitGradientOp<Context>::DoRunWithType() {
  auto* dX = Output(0);

  int num_splits = InputSize();
  CANONICALIZE_AXIS_WITH_TENSOR((*dX));
  DETERMINE_RUNTIME_ARGS((*dX));

  // Zero the missing gradients if necessary
  for (int i = 0; i < num_splits; i++) {
    if (!Input(i).has_name()) {
      math::Set(
          dX->count(),
          cast::to<T>(0.f),
          dX->template mutable_data<T, Context>(),
          ctx());
      break;
    }
  }

  int64_t index = 0;
  for (int i = 0; i < num_splits; i++) {
    auto& dY = Input(i);
    if (dY.has_name()) {
      kernel::Concat(
          dX->count(0, axis),
          dX->count(axis + 1),
          size_splits[i],
          dX->dim(axis),
          index,
          dY.template data<T, Context>(),
          dX->template mutable_data<T, Context>(),
          ctx());
    }
    index += size_splits[i];
  }
}

template <class Context>
void SplitGradientOp<Context>::RunOnDevice() {
  auto& X = RESTORE_INPUT_SPEC(0);
  Output(0)->ReshapeLike(X);
  DispatchHelper<FloatingTensorTypes>::Call(this, X);
}

DEPLOY_CPU_OPERATOR(Split);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Split);
#endif

DEPLOY_CPU_OPERATOR(SplitGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SplitGradient);
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

#undef DETERMINE_RUNTIME_ARGS

} // namespace dragon
