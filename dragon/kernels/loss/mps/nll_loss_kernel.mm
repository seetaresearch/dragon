#include "dragon/kernels/loss/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // C
constant uint uint_arg2 [[function_constant(1)]]; // S
constant int int_arg1 [[function_constant(2)]]; // ignore_index

template <typename InputT, typename TargetT>
kernel void NLLLoss(
    device const InputT* input,
    device const TargetT* target,
    device InputT* loss,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index / uint_arg2, j = index % uint_arg2;
  const int k = int(target[index]);
   if (k == int_arg1) {
    loss[index] = InputT(0);
  } else {
    loss[index] = -input[(i * uint_arg1 + k) * uint_arg2 + j];
  }
}

template <typename InputT, typename TargetT>
kernel void NLLLossGrad(
    device const InputT* input, /* not used */
    device const TargetT* target,
    device InputT* dx,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index / uint_arg2, j = index % uint_arg2;
  const int k = int(target[index]);
  if (k == int_arg1) {
  } else {
    dx[(i * uint_arg1 + k) * uint_arg2 + j] = InputT(-1);
  }
}

#define INSTANTIATE_KERNEL(name, InputT, TargetT) \
  template [[host_name(#name"_"#InputT"_"#TargetT)]] \
  kernel void name(device const InputT*, device const TargetT*, \
                   device InputT*, uint);

INSTANTIATE_KERNEL(NLLLoss, float, int);
INSTANTIATE_KERNEL(NLLLoss, float, int64_t);
INSTANTIATE_KERNEL(NLLLossGrad, float, int);
INSTANTIATE_KERNEL(NLLLossGrad, float, int64_t);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(NLLLoss, double, int);
INSTANTIATE_KERNEL(NLLLoss double, int64_t);
INSTANTIATE_KERNEL(NLLLossGrad, double, int);
INSTANTIATE_KERNEL(NLLLossGrad, double, int64_t);
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
      const int ignore_index,                                         \
      const InputT* input,                                            \
      const TargetT* target,                                          \
      InputT* loss,                                                   \
      InputT* mask,                                                   \
      MPSContext* ctx) {                                              \
    const uint arg1 = C, arg2 = S;                                    \
    auto kernel = MPSKernel::TypedString<InputT>(#name);              \
    kernel = MPSKernel::TypedString<TargetT>(kernel);                 \
    auto args = vector<MPSConstant>(                                  \
        {MPSConstant(&arg1, MTLDataTypeUInt, 0),                      \
         MPSConstant(&arg2, MTLDataTypeUInt, 1),                      \
         MPSConstant(&ignore_index, MTLDataTypeInt, 2)});             \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(input) offset:0 atIndex:0];      \
    [encoder setBuffer:id<MTLBuffer>(target) offset:0 atIndex:1];     \
    [encoder setBuffer:id<MTLBuffer>(loss) offset:0 atIndex:2];       \
    MPSDispatchThreads((N * S), encoder, pso);                        \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(NLLLoss, float, int);
DEFINE_KERNEL_LAUNCHER(NLLLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, int);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, int);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, int);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
