#include "dragon/operators/math/topk_op.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSTopKOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y_value = Output(0), *Y_index = Output(1);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  CHECK_LE(k_, C) << "\nThe top-K argument is out of the dimension.";
  auto Y_dims = X.dims();
  Y_dims[axis] = k_;

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        auto* X_value = largest_ > 0
            ? placeholders[0]
            : [graph_ negativeWithTensor:placeholders[0] name:nil];
        if (axis != X.ndim() - 1) {
          X_value = [graph_ transposeTensor:X_value
                                  dimension:X.ndim() - 1
                              withDimension:axis
                                       name:nil];
          // Add "Identity" to derive the correct index tensor.
          X_value = [graph_ identityWithTensor:X_value name:nil];
        }
        auto* Ys = [graph_ topKWithSourceTensor:X_value k:k_ name:nil];
        placeholders.emplace_back(Ys[0]);
        placeholders.emplace_back(Ys[1]);
        for (int i = 1; i <= 2 && (axis != X.ndim() - 1); ++i) {
          placeholders[i] = [graph_ transposeTensor:placeholders[i]
                                          dimension:X.ndim() - 1
                                      withDimension:axis
                                               name:nil];
        }
        placeholders[1] = largest_ > 0
            ? placeholders[1]
            : [graph_ negativeWithTensor:placeholders[1] name:nil];
        placeholders[2] = [graph_ castTensor:placeholders[2]
                                      toType:MPSDataTypeInt64
                                        name:(NSString* _Nonnull)nil];
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y_value->Reshape(Y_dims)->template mutable_data<T, Context>(),
          placeholders[1]),
      placeholders[2] : MPSCreateTensorData(
          Y_index->Reshape(Y_dims)->template mutable_data<int64_t, Context>(),
          placeholders[2]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(TopK, MPSTopK);

} // namespace dragon
