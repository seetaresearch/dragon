#include "dragon/operators/math/arg_op.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSArgMaxOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  auto Y_dims = X.dims();
  if (!keep_dims_) {
    Y_dims.erase(Y_dims.begin() + axis);
  } else {
    Y_dims[axis] = 1;
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        auto* Y_int32 = [graph_ reductionArgMaximumWithTensor:placeholders[0]
                                                         axis:axis
                                                         name:nil];
        placeholders.emplace_back([graph_ castTensor:Y_int32
                                              toType:MPSDataTypeInt64
                                                name:(NSString* _Nonnull)nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
          placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSArgMinOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), 0);

  auto Y_dims = X.dims();
  if (!keep_dims_) {
    Y_dims.erase(Y_dims.begin() + axis);
  } else {
    Y_dims[axis] = 1;
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        auto* Y_int32 = [graph_ reductionArgMinimumWithTensor:placeholders[0]
                                                         axis:axis
                                                         name:nil];
        placeholders.emplace_back([graph_ castTensor:Y_int32
                                              toType:MPSDataTypeInt64
                                                name:(NSString* _Nonnull)nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
          placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(ArgMax, MPSArgMax);
DEPLOY_MPS_OPERATOR(ArgMin, MPSArgMin);

} // namespace dragon
