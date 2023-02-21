#include "dragon/operators/activation/dropout_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSDropoutOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    const auto N = X.count();
    const auto X_dims = vec64_t({N});
    const auto seed = ctx()->random_seed();
    const auto drop_ratio = this->ratio();
    const auto state_name = "MPSPhiloxState:" + str::to(seed);
    auto* X_mask = Output("X_mask")->ReshapeLike(X);
    auto* Y_state = ctx()->workspace()->CreateTensor(state_name);
    auto* Y_state_inc = ctx()->workspace()->GetTensor("MPSPhiloxStateInc");
    // Create philox state.
    if (Y_state->empty()) {
      ctx()->mps_stream()->CreatePhiloxState(
          graph_,
          seed,
          Y_state->Reshape({7})->template mutable_data<int, Context>());
    }
    // Generate radnom values.
    auto placeholders = graph_cache_.GetPlaceholders(
        {&X_dims}, {}, [&](vector<MPSGraphTensor_t>& placeholders) {
          auto* desc = [[MPSGraphRandomOpDescriptor new] autorelease];
          desc.dataType = MPSGetDataType(TypeMeta::Make<float>());
          desc.distribution = MPSGraphRandomDistributionUniform;
          desc.min = 0.f, desc.max = 1.f;
          placeholders.emplace_back(MPSCreateTensor<int>(graph_, {7}));
          auto* results = [graph_ randomTensorWithShape:MPSGetShape(X_dims)
                                             descriptor:desc
                                            stateTensor:placeholders[0]
                                                   name:nil];
          placeholders.emplace_back(results[0]);
        });
    auto* scratch = ctx()->workspace()->template data<float, Context>(N);
    @autoreleasepool {
      auto* inputs = @{
        placeholders[0] : MPSCreateTensorData(
            Y_state->template data<int, Context>(), placeholders[0]),
      };
      auto* outputs = @{
        placeholders[1] : MPSCreateTensorData(scratch, placeholders[1]),
      };
      ctx()->mps_stream()->Encode(graph_, inputs, outputs);
    }
    // Apply Dropout.
    kernels::Dropout(
        X.count(),
        drop_ratio,
        1.f / (1.f - drop_ratio),
        scratch,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        X_mask->template mutable_data<uint8_t, Context>(),
        ctx());
    // Update philox state.
    math::Add(
        4,
        Y_state->template data<int, Context>(),
        Y_state_inc->template data<int, Context>(),
        Y_state->template mutable_data<int, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

DEPLOY_MPS_OPERATOR(Dropout, MPSDropout);

DEFINE_OP_SINGLE_ARG(float, MPSDropoutOp, ratio);

} // namespace dragon
