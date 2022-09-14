#include "dragon/operators/activation/dropout_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSDropoutOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    const auto N = X.count();
    const auto drop_ratio = ratio();
    const auto seed = vec64_t({graph_seeds_.front()});
    graph_seeds_.push(seed[0]);
    graph_seeds_.pop();
    auto* X_mask = Output("X_mask")->ReshapeLike(X);
    auto* scratch = ctx()->workspace()->template data<float, Context>(N);
    auto placeholders = graph_cache_.GetPlaceholders(
        {&X_mask->dims(), &seed},
        {},
        [&](vector<MPSGraphTensor_t>& placeholders) {
          placeholders.emplace_back([graph_
              randomUniformTensorWithShape:MPSGetShape({N})
                                      seed:seed[0]
                                      name:nil]);
        });
    @autoreleasepool {
      auto* outputs = @{
        placeholders[0] : MPSCreateTensorData(scratch, placeholders[0]),
      };
      ctx()->mps_stream()->Encode(graph_, @{}, outputs);
    }
    kernels::Dropout(
        X.count(),
        drop_ratio,
        1.f / (1.f - drop_ratio),
        scratch,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        X_mask->template mutable_data<uint8_t, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

DEPLOY_MPS_OPERATOR(Dropout, MPSDropout);

} // namespace dragon
