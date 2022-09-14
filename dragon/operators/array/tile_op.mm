#include "dragon/operators/array/tile_op.h"
#include "dragon/core/workspace.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSTileOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_repeats, num_dims, start_axis;
  repeats(0, &num_repeats);
  num_dims = std::max(num_repeats, X.ndim());
  start_axis = num_dims - X.ndim();

  vec64_t X_dims(num_dims, 1), X_reps(num_dims, 1), Y_dims(num_dims, 1);
  std::copy(X.dims().begin(), X.dims().end(), X_dims.begin() + start_axis);
  std::copy(X.dims().begin(), X.dims().end(), Y_dims.begin() + start_axis);
  for (int i = 0; i < num_repeats; ++i) {
    start_axis = i + (num_dims - num_repeats);
    X_reps[start_axis] = repeats(i);
    Y_dims[start_axis] = X_dims[start_axis] * X_reps[start_axis];
  }

  // Save for the gradient computation.
  Output("X_spec")->ReshapeLike(X);
  Output("X_dims")->template CopyFrom<int64_t>(X_dims);
  Output("Y_dims")->template CopyFrom<int64_t>(Y_dims);

  if (X_dims == Y_dims) {
    Y->Reshape(Y_dims)->CopyFrom(X, ctx());
    return;
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &Y_dims},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        placeholders.emplace_back([graph_ tileTensor:placeholders[0]
                                      withMultiplier:MPSGetShape(X_reps)
                                                name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
          placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSTileGradientOp<Context>::DoRunWithType() {
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

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X_dims, &Y_dims},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, Y_reduce_dims));
        placeholders.emplace_back([graph_
            reductionSumWithTensor:placeholders[0]
                              axes:MPSGetShape(Y_reduce_axes)
                              name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          dX->template mutable_data<T, Context>(), placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(Tile, MPSTile);
DEPLOY_MPS_OPERATOR(TileGradient, MPSTileGradient);

} // namespace dragon
