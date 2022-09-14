#include "dragon/operators/array/reverse_op.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSReverseOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        placeholders.emplace_back([graph_ reverseTensor:placeholders[0]
                                                   axes:MPSGetShape(axes_)
                                                   name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y->ReshapeLike(X)->template mutable_data<T, Context>(),
          placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(Reverse, MPSReverse);
REGISTER_MPS_OPERATOR(ReverseGradient, MPSReverseOp<MPSContext>);

} // namespace dragon
