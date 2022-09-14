#include "dragon/operators/array/gather_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSGatherOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_index = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);

  CHECK_GT(X_index.count(), 0) << "\nLength of index must > 0.";
  vec64_t X_dims(X.dims()), Y_dims;
  vec64_t Y_shape(X_dims.begin(), X_dims.begin() + axis);
  Y_shape.insert(Y_shape.end(), X_index.dims().begin(), X_index.dims().end());
  Y_shape.insert(Y_shape.end(), X_dims.begin() + end_axis + 1, X_dims.end());
  const auto C1 = X.count(axis, end_axis + 1), C2 = X_index.count();
  X_dims = {X.count(0, axis), C1, X.count(end_axis + 1)};
  Y_dims = {X.count(0, axis), C2, X.count(end_axis + 1)};

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X_dims, &Y_dims},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X_dims));
        placeholders.emplace_back(MPSCreateTensor<int64_t>(graph_, {C2}));
        placeholders.emplace_back([graph_
            gatherWithUpdatesTensor:placeholders[0]
                      indicesTensor:placeholders[1]
                               axis:1
                    batchDimensions:0
                               name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
      placeholders[1] : MPSCreateTensorData(
          X_index.template data<int64_t, Context>(), placeholders[1]),
    };
    auto* outputs = @{
      placeholders[2] : MPSCreateTensorData(
          Y->Reshape(Y_shape)->template mutable_data<T, Context>(),
          placeholders[2]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSGatherGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(Input("X_spec"));
  GET_OP_AXIS_ARG(axis, dX->ndim(), 0);
  GET_OP_AXIS_ARG(end_axis, dX->ndim(), axis);

  const auto C1 = dX->count(axis, end_axis + 1), C2 = X_index.count();
  vec64_t X_dims = {dX->count(0, axis), C1, dX->count(end_axis + 1)};
  vec64_t Y_dims = {dX->count(0, axis), C2, dX->count(end_axis + 1)};

  math::Set(
      dX->count(),
      convert::To<T>(0.f),
      dX->template mutable_data<T, Context>(),
      ctx());

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X_dims, &Y_dims},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, Y_dims));
        placeholders.emplace_back(MPSCreateTensor<int64_t>(graph_, {C2}));
        placeholders.emplace_back([graph_
            scatterWithDataTensor:placeholders[0]
                    updatesTensor:placeholders[1]
                    indicesTensor:placeholders[2]
                             axis:1
                             mode:MPSGraphScatterModeAdd
                             name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(dX->template data<T, Context>(), placeholders[0]),
      placeholders[1] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[1]),
      placeholders[2] : MPSCreateTensorData(
          X_index.template data<int64_t, Context>(), placeholders[2]),
    };
    auto* outputs = @{
      placeholders[3] : MPSCreateTensorData(
          dX->template mutable_data<T, Context>(), placeholders[3]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(Gather, MPSGather);
DEPLOY_MPS_OPERATOR(GatherGradient, MPSGatherGradient);

} // namespace dragon
