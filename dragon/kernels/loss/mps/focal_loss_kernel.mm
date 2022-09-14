#include "dragon/kernels/loss/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]];   // C
constant uint uint_arg2 [[function_constant(1)]];   // S
constant int int_arg1 [[function_constant(2)]];     // start_index
constant float float_arg1 [[function_constant(3)]]; // alpha
constant float float_arg2 [[function_constant(4)]]; // gamma

template <typename InputT, typename TargetT>
kernel void SigmoidFocalLoss(
    device const InputT* input,
    device const TargetT* target,
    device InputT* loss,
    const uint index [[thread_position_in_grid]]) {
  const uint j = index % uint_arg2;
  const uint k = index / uint_arg2 % uint_arg1;
  const uint i = index / (uint_arg2 * uint_arg1);
  const float lgt = float(input[index]);
  const int tgt = target[i * uint_arg2 + j];
  const float c1 = float(tgt == (k + int_arg1));
  const float c2 = float((tgt >= 0) & (tgt != (k + int_arg1)));
  const float p = 1.f / (1.f + exp(-lgt));
  // pos_term: (1 - p)^{gamma} * log(p)
  float v1 = pow(1.f - p, float_arg2) * log(max(p, FLT_MIN));
  // p^{gamma} * log(1 - p)
  const float v2 = pow(p, float_arg2) *
      (-lgt * (lgt >= 0.f) - log(1.f + exp(lgt - 2.f * lgt * (lgt >= 0.f))));
  loss[index] = -(c1 * v1 * float_arg1 + c2 * v2 * (1.f - float_arg1));
}

template <typename InputT, typename TargetT>
kernel void SigmoidFocalLossGrad(
    device const InputT* input,
    device const TargetT* target,
    device InputT* dx,
    const uint index [[thread_position_in_grid]]) {
  const uint j = index % uint_arg2;
  const uint k = index / uint_arg2 % uint_arg1;
  const uint i = index / (uint_arg2 * uint_arg1);
  const float lgt = float(input[index]);
  const int tgt = target[i * uint_arg2 + j];
  const float c1 = float(tgt == (k + int_arg1));
  const float c2 = float((tgt >= 0) & (tgt != (k + int_arg1)));
  const float p = 1.f / (1.f + exp(-lgt));
  // pos_term: (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
  const float v1 = pow(1.f - p, float_arg2) *
      (1.f - p - float_arg2 * p * log(max(p, FLT_MIN)));
  // neg_term: p^{gamma} * (gamma * (1 - p) * log(1 - p) - p)
  const float v2 = pow(p, float_arg2) *
      ((-lgt * (lgt >= 0.f) - log(1.f + exp(lgt - 2.f * lgt * (lgt >= 0.f))))
          * (1.f - p) * float_arg2 - p);
  dx[index] = -(c1 * v1 * float_arg1 + c2 * v2 * (1.f - float_arg1));
}

#define INSTANTIATE_KERNEL(name, InputT, TargetT) \
  template [[host_name(#name"_"#InputT"_"#TargetT)]] \
  kernel void name(device const InputT*, device const TargetT*, \
                   device InputT*, uint);

INSTANTIATE_KERNEL(SigmoidFocalLoss, float, int);
INSTANTIATE_KERNEL(SigmoidFocalLoss, float, int64_t);
INSTANTIATE_KERNEL(SigmoidFocalLossGrad, float, int);
INSTANTIATE_KERNEL(SigmoidFocalLossGrad, float, int64_t);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(SigmoidFocalLoss, double, int);
INSTANTIATE_KERNEL(SigmoidFocalLoss, double, int64_t);
INSTANTIATE_KERNEL(SigmoidFocalLossGrad, double, int);
INSTANTIATE_KERNEL(SigmoidFocalLossGrad, double, int64_t);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT)                 \
  template <>                                                         \
  void name<InputT, TargetT, MPSContext>(                             \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const int start_index,                                          \
      const float alpha,                                              \
      const float gamma,                                              \
      const InputT* input,                                            \
      const TargetT* target,                                          \
      InputT* loss,                                                   \
      InputT* mask,                                                   \
      MPSContext* ctx) {                                              \
    const uint arg1 = C, arg2 = S;                                    \
    auto kernel = MPSKernel::TypedString<InputT>(#name);              \
    kernel = MPSKernel::TypedString<TargetT>(kernel);                 \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                       \
        MPSConstant(&arg2, MTLDataTypeUInt, 1),                       \
        MPSConstant(&start_index, MTLDataTypeInt, 2),                 \
        MPSConstant(&alpha, MTLDataTypeFloat, 3),                     \
        MPSConstant(&gamma, MTLDataTypeFloat, 4),                     \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(input) offset:0 atIndex:0];      \
    [encoder setBuffer:id<MTLBuffer>(target) offset:0 atIndex:1];     \
    [encoder setBuffer:id<MTLBuffer>(loss) offset:0 atIndex:2];       \
    MPSDispatchThreads((N * C * S), encoder, pso);                    \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
