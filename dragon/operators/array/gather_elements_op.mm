#include "dragon/core/workspace.h"
#include "dragon/operators/array/gather_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSGatherElementsOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_index = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  CHECK_GT(X_index.count(), 0) << "\nLength of index must > 0.";
  CHECK_EQ(X.ndim(), X_index.ndim())
      << "\nMismatched number of dimensions between input and index.";
  for (int i = 0; i < X.ndim(); ++i) {
    if (i != axis) CHECK_EQ(X_index.dim(i), X.dim(i));
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &X_index.dims()},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        // clang-format off
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
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
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
      placeholders[1] : MPSCreateTensorData(
          X_index.template data<int64_t, Context>(), placeholders[1]),
    };
    auto* outputs = @{
      placeholders[2] : MPSCreateTensorData(
          Y->ReshapeLike(X_index)->template mutable_data<T, Context>(),
          placeholders[2]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSGatherElementsGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto &X_spec = Input("X_spec"), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, X_spec.ndim(), 0);

  math::Set(
      dX->count(),
      convert::To<T>(0.f),
      dX->ReshapeLike(X_spec)->template mutable_data<T, Context>(),
      ctx());

  auto placeholders = graph_cache_.GetPlaceholders(
      {&dX->dims(), &X_index.dims()},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        // clang-format off
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dX->dims()));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY.dims()));
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

DEPLOY_MPS_OPERATOR(GatherElements, MPSGatherElements);
DEPLOY_MPS_OPERATOR(GatherElementsGradient, MPSGatherElementsGradient);

} // namespace dragon
