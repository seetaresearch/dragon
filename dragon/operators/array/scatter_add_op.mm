#include "dragon/operators/array/scatter_op.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSScatterAddOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto &X_index = Input(1), &X_value = Input(2);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  CHECK_GT(X_index.count(), 0) << "\nLength of index must > 0.";
  CHECK_EQ(X.ndim(), X_index.ndim())
      << "\nMismatched number of dimensions between input and index.";
  CHECK_EQ(X_index.ndim(), X_value.ndim())
      << "\nMismatched number of dimensions between index and value.";
  for (int i = 0; i < X.ndim(); ++i) {
    CHECK_LE(X_index.dim(i), X_value.dim(i));
    if (i != axis) CHECK_LE(X_index.dim(i), X_value.dim(i));
  }

  // Copy the input data.
  Y->ReshapeLike(X)->CopyFrom(X, ctx());

  // Update with the new data.
  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &X_index.dims()},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        // clang-format off
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, Y->dims()));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X_value.dims()));
        placeholders.emplace_back(MPSCreateTensor<int64_t>(graph_, X_index.dims()));
        auto* index32 = [graph_ castTensor:placeholders[2]
                                    toType:MPSGetDataType(TypeMeta::Make<int>())
                                      name:(NSString* _Nonnull)nil];
        placeholders.emplace_back([graph_ scatterAlongAxis:axis
                                            withDataTensor:placeholders[0]
                                             updatesTensor:placeholders[1]
                                            indicesTensor:index32
                                                     mode:MPSGraphScatterModeAdd
                                                     name:nil]); // clang-format on
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(Y->template data<T, Context>(), placeholders[0]),
      placeholders[1] : MPSCreateTensorData(
          X_value.template data<T, Context>(), placeholders[1]),
      placeholders[2] : MPSCreateTensorData(
          X_index.template data<int64_t, Context>(), placeholders[2]),
    };
    auto* outputs = @{
      placeholders[3] : MPSCreateTensorData(
          Y->template mutable_data<T, Context>(), placeholders[3]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSScatterAddGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto *dX = Output(0), *dX_value = Output(1);
  GET_OP_AXIS_ARG(axis, dY.ndim(), 0);

  if (dX->has_name()) {
    dX->ReshapeLike(dY)->CopyFrom(dY, ctx());
  }

  if (!dX_value->has_name()) return;

  auto placeholders = graph_cache_.GetPlaceholders(
      {&dY.dims(), &X_index.dims()},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        // clang-format off
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY.dims()));
        placeholders.emplace_back(MPSCreateTensor<int64_t>(graph_, X_index.dims()));
        auto* index32 = [graph_ castTensor:placeholders[1]
                                    toType:MPSGetDataType(TypeMeta::Make<int>())
                                      name:(NSString* _Nonnull)nil];
        placeholders.emplace_back([graph_ gatherAlongAxis:axis
                                        withUpdatesTensor:placeholders[0]
                                            indicesTensor:index32
                                                     name:nil]); // clang-format on
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[0]),
      placeholders[1] : MPSCreateTensorData(
          X_index.template data<int64_t, Context>(), placeholders[1]),
    };
    auto* outputs = @{
      placeholders[2] : MPSCreateTensorData(
          dX_value->ReshapeLike(X_index)->template mutable_data<T, Context>(),
          placeholders[2]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(ScatterAdd, MPSScatterAdd);
DEPLOY_MPS_OPERATOR(ScatterAddGradient, MPSScatterAddGradient);

} // namespace dragon
