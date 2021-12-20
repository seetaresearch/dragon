#include "dragon/operators/array/tile_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void TileOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_repeats, num_dims, start_axis;
  repeats(0, &num_repeats);
  num_dims = std::max(num_repeats, X.ndim());
  start_axis = num_dims - X.ndim();

  vec64_t X_dims(num_dims, 1), Y_dims(num_dims, 1);
  std::copy(X.dims().begin(), X.dims().end(), X_dims.begin() + start_axis);
  std::copy(X.dims().begin(), X.dims().end(), Y_dims.begin() + start_axis);
  for (int i = 0; i < num_repeats; ++i) {
    start_axis = i + (num_dims - num_repeats);
    Y_dims[start_axis] = X_dims[start_axis] * repeats(i);
  }

  // Save for the gradient computation.
  Output("X_spec")->ReshapeLike(X);
  Output("X_dims")->template CopyFrom<int64_t>(X_dims);
  Output("Y_dims")->template CopyFrom<int64_t>(Y_dims);

  if (X_dims == Y_dims) {
    Y->Reshape(Y_dims)->CopyFrom(X, ctx());
    return;
  }

  kernels::Tile(
      num_dims,
      X_dims.data(),
      Tensor(X_dims).strides().data(),
      Y_dims.data(),
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void TileGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  vec64_t X_dims, Y_dims;
  vec64_t Y_reduce_axes, Y_reduce_dims;
  Input("X_dims").template CopyTo<int64_t>(X_dims);
  Input("Y_dims").template CopyTo<int64_t>(Y_dims);
  for (int i = 0; i < X_dims.size(); ++i) {
    if (X_dims[i] != Y_dims[i]) {
      Y_reduce_axes.push_back(int64_t(Y_reduce_dims.size()));
      Y_reduce_dims.push_back(Y_dims[i] / X_dims[i]);
    }
    if (X_dims[i] != 1) {
      Y_reduce_dims.push_back(X_dims[i]);
    }
  }

  if (Y_reduce_axes.empty()) {
    dX->CopyFrom(dY, ctx());
    return;
  }

  math::ReduceSum(
      Y_reduce_dims.size(),
      Y_reduce_dims.data(),
      Y_reduce_axes.size(),
      Y_reduce_axes.data(),
      1.f,
      dY.template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(Tile);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Tile);
#endif

DEPLOY_CPU_OPERATOR(TileGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(TileGradient);
#endif

OPERATOR_SCHEMA(Tile)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(TileGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Tile, SimpleGradientMaker);

} // namespace dragon
