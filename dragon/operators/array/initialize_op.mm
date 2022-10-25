#include "dragon/operators/array/initialize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSRandomOpBase<Context>::DoRunWithType() {
  auto* Y = Output(0);
  const auto N = Y->count();
  const auto Y_dims = vec64_t({N});
  const auto Y_meta = dtypes::to_meta(data_type());
  const auto seed = def().device_option().random_seed();
  const auto state_name = "MPSPhiloxState:" + str::to(seed);
  auto* Y_state = ctx()->workspace()->CreateTensor(state_name);
  auto* Y_state_inc = ctx()->workspace()->GetTensor("MPSPhiloxStateInc");

  // Create philox state.
  if (Y_state->empty()) {
    ctx()->mps_stream()->CreatePhiloxState(
        graph_,
        seed,
        Y_state->Reshape({7})->template mutable_data<int, Context>());
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&Y_dims}, {&Y_meta}, [&](vector<MPSGraphTensor_t>& placeholders) {
        auto* desc = [[MPSGraphRandomOpDescriptor new] autorelease];
        desc.dataType = MPSGetDataType(Y_meta);
        SetOpDesc(desc);
        placeholders.emplace_back(MPSCreateTensor<int>(graph_, {7}));
        auto* results = [graph_ randomTensorWithShape:MPSGetShape(Y_dims)
                                           descriptor:desc
                                          stateTensor:placeholders[0]
                                                 name:nil];
        placeholders.emplace_back(results[0]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] : MPSCreateTensorData(
          Y_state->template data<int, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y->template mutable_data<T, Context>(), placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }

  // Update philox state.
  math::Add(
      4,
      Y_state->template data<int, Context>(),
      Y_state_inc->template data<int, Context>(),
      Y_state->template mutable_data<int, Context>(),
      ctx());
}

template <class Context>
void MPSRandomUniformOp<Context>::SetOpDesc(
    MPSGraphRandomOpDescriptor_t op_desc) {
  op_desc.distribution = MPSGraphRandomDistributionUniform;
  op_desc.min = low_, op_desc.max = high_;
}

template <class Context>
void MPSRandomNormalOp<Context>::SetOpDesc(
    MPSGraphRandomOpDescriptor_t op_desc) {
  op_desc.distribution = MPSGraphRandomDistributionNormal;
  op_desc.mean = mean_, op_desc.standardDeviation = std_;
}

template <class Context>
void MPSTruncatedNormalOp<Context>::SetOpDesc(
    MPSGraphRandomOpDescriptor_t op_desc) {
  op_desc.distribution = MPSGraphRandomDistributionTruncatedNormal;
  op_desc.min = low_, op_desc.max = high_;
  op_desc.mean = mean_, op_desc.standardDeviation = std_;
}

DEPLOY_MPS_OPERATOR(RandomUniform, MPSRandomUniform);
DEPLOY_MPS_OPERATOR(RandomNormal, MPSRandomNormal);
DEPLOY_MPS_OPERATOR(TruncatedNormal, MPSTruncatedNormal);

} // namespace dragon
