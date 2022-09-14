#include "dragon/operators/array/initialize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

#define DISPATCH_FOR_DTYPES(name, dtypes) \
  template <class Context>                \
  void name##Op<Context>::RunOnDevice() { \
    InitializeOp<Context>::RunOnDevice(); \
    DispatchHelper<dtypes>::Call(this);   \
  }

DISPATCH_FOR_DTYPES(MPSRandomUniform, dtypes::Floating);
DISPATCH_FOR_DTYPES(MPSRandomNormal, dtypes::Floating);
DISPATCH_FOR_DTYPES(MPSTruncatedNormal, dtypes::Floating);
#undef DISPATCH_FOR_DTYPES

template <class Context>
template <typename T>
void MPSRandomUniformOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  const auto N = Y->count();
  const auto seed = static_cast<uint32_t>((*ctx()->rand_generator())());
  auto* data = Y->template mutable_data<T, Context>();
  if (@available(macOS 12.3, *)) {
    @autoreleasepool {
      auto* desc = [[MPSGraphRandomOpDescriptor new] autorelease];
      desc.dataType = MPSGetDataType(Y->meta());
      desc.distribution = MPSGraphRandomDistributionUniform;
      desc.min = low_, desc.max = high_;
      auto* placeholder = [graph_ randomTensorWithShape:MPSGetShape({N})
                                             descriptor:desc
                                                   seed:seed
                                                   name:nil];
      auto* outputs = @{
        placeholder : MPSCreateTensorData(data, placeholder),
      };
      ctx()->mps_stream()->Encode(graph_, @{}, outputs);
    }
  } else {
    auto* scratch = ctx()->workspace()->template data<float, Context>(N);
    @autoreleasepool {
      auto* placeholder = [graph_ randomUniformTensorWithShape:MPSGetShape({N})
                                                          seed:seed
                                                          name:nil];
      auto* outputs = @{
        placeholder : MPSCreateTensorData(scratch, placeholder),
      };
      ctx()->mps_stream()->Encode(graph_, @{}, outputs);
    }
    math::Cast(N, scratch, data, ctx());
    math::Scale(N, high_ - low_, data, data, ctx());
    math::Bias(N, low_, data, data, ctx());
  }
}

template <class Context>
template <typename T>
void MPSRandomNormalOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  const auto N = Y->count();
  const auto seed = static_cast<uint32_t>((*ctx()->rand_generator())());
  auto* data = Y->template mutable_data<T, Context>();
  @autoreleasepool {
    auto* desc = [[MPSGraphRandomOpDescriptor new] autorelease];
    desc.dataType = MPSGetDataType(Y->meta());
    desc.distribution = MPSGraphRandomDistributionNormal;
    desc.mean = mean_, desc.standardDeviation = std_;
    auto* placeholder = [graph_ randomTensorWithShape:MPSGetShape({N})
                                           descriptor:desc
                                                 seed:seed
                                                 name:nil];
    auto* outputs = @{
      placeholder : MPSCreateTensorData(data, placeholder),
    };
    ctx()->mps_stream()->Encode(graph_, @{}, outputs);
  }
}

template <class Context>
template <typename T>
void MPSTruncatedNormalOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  const auto N = Y->count();
  const auto seed = static_cast<uint32_t>((*ctx()->rand_generator())());
  auto* data = Y->template mutable_data<T, Context>();
  @autoreleasepool {
    auto* desc = [[MPSGraphRandomOpDescriptor new] autorelease];
    desc.dataType = MPSGetDataType(Y->meta());
    desc.distribution = MPSGraphRandomDistributionTruncatedNormal;
    desc.min = low_, desc.max = high_;
    desc.mean = mean_, desc.standardDeviation = std_;
    auto* placeholder = [graph_ randomTensorWithShape:MPSGetShape({N})
                                           descriptor:desc
                                                 seed:seed
                                                 name:nil];
    auto* outputs = @{
      placeholder : MPSCreateTensorData(data, placeholder),
    };
    ctx()->mps_stream()->Encode(graph_, @{}, outputs);
  }
}

DEPLOY_MPS_OPERATOR(RandomUniform, MPSRandomUniform);
DEPLOY_MPS_OPERATOR(RandomNormal, MPSRandomNormal);
DEPLOY_MPS_OPERATOR(TruncatedNormal, MPSTruncatedNormal);

} // namespace dragon
