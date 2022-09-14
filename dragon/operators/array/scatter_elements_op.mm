#include "dragon/operators/array/scatter_op.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSScatterElementsOp<Context>::DoRunWithType() {
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
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, Y->dims()));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X_value.dims()));
        placeholders.emplace_back(
            MPSCreateTensor<int64_t>(graph_, X_index.dims()));
        auto* index32 = [graph_ castTensor:placeholders[2]
                                    toType:MPSGetDataType(TypeMeta::Make<int>())
                                      name:(NSString* _Nonnull)nil];
        placeholders.emplace_back([graph_
            scatterAlongAxis:axis
              withDataTensor:placeholders[0]
               updatesTensor:placeholders[1]
               indicesTensor:index32
                        mode:MPSGraphScatterModeSet
                        name:nil]);
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
void MPSScatterElementsGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto *dX = Output(0), *dX_value = Output(1);
  GET_OP_AXIS_ARG(axis, dY.ndim(), 0);

  auto placeholders = graph_cache_.GetPlaceholders(
      {&dY.dims(), &X_index.dims()},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        // clang-format off
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY.dims()));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY.dims()));
        placeholders.emplace_back(MPSCreateTensor<int64_t>(graph_, X_index.dims()));
        auto* index32 = [graph_ castTensor:placeholders[2]
                                    toType:MPSGetDataType(TypeMeta::Make<int>())
                                      name:(NSString* _Nonnull)nil];
        auto* dX_fill = [graph_ constantWithScalar:0.
                                          shape:MPSGetShape(X_index.dims())
                                          dataType:MPSGetDataType(TypeMeta::Make<T>())];
        placeholders.emplace_back([graph_ scatterAlongAxis:axis
                                            withDataTensor:placeholders[0]
                                             updatesTensor:dX_fill
                                             indicesTensor:index32
                                                      mode:MPSGraphScatterModeSet
                                                      name:nil]);
        placeholders.emplace_back([graph_ gatherAlongAxis:axis
                                        withUpdatesTensor:placeholders[1]
                                            indicesTensor:index32
                                                     name:nil]);
      });

  // No @autoreleasepool here to avoid crash.
  auto* inputs = [[[NSMutableDictionary alloc] init] autorelease];
  auto* outputs = [[[NSMutableDictionary alloc] init] autorelease];
  if (dX->has_name()) {
    dX->ReshapeLike(dY)->CopyFrom(dY, ctx());
    inputs[placeholders[0]] = MPSCreateTensorData(
        dX->template data<T, Context>(), placeholders[0]);
   }
  inputs[placeholders[1]] = MPSCreateTensorData(
      dY.template data<T, Context>(), placeholders[1]);
  inputs[placeholders[2]] = MPSCreateTensorData(
      X_index.template data<int64_t, Context>(), placeholders[2]);
  // clang-format on
  if (dX->has_name()) {
    outputs[placeholders[3]] = MPSCreateTensorData(
        dX->template mutable_data<T, Context>(), placeholders[3]);
  }
  if (dX_value->has_name()) {
    outputs[placeholders[4]] = MPSCreateTensorData(
        dX_value->ReshapeLike(X_index)->template mutable_data<T, Context>(),
        placeholders[4]);
  }
  ctx()->mps_stream()->Encode(graph_, inputs, outputs);
}

DEPLOY_MPS_OPERATOR(ScatterElements, MPSScatterElements);
DEPLOY_MPS_OPERATOR(ScatterElementsGradient, MPSScatterElementsGradient);

} // namespace dragon
