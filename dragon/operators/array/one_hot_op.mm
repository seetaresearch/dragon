#include "dragon/operators/array/one_hot_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSOneHotOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  vec64_t Y_dims(X.dims());
  Y_dims.push_back(depth_);

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        placeholders.emplace_back([graph_
            oneHotWithIndicesTensor:placeholders[0]
                              depth:depth_
                               axis:-1
                           dataType:MPSGetDataType(X.meta())
                            onValue:on_value_
                           offValue:off_value_
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

DEPLOY_MPS_OPERATOR(OneHot, MPSOneHot);

} // namespace dragon
