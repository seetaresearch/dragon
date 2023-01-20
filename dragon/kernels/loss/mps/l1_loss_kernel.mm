#include "dragon/kernels/loss/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]]; // beta

template <typename T>
kernel void SmoothL1Loss(
    device const T* input,
    device const T* target,
    device T* loss,
    const uint index [[thread_position_in_grid]]) {
  const T val = input[index] - target[index];
  const T abs_val = abs(val);
  loss[index] = abs_val < T(float_arg1) ? T(.5) * val * val / T(float_arg1)
                                        : abs_val - T(.5) * T(float_arg1);
}

template <typename T>
kernel void SmoothL1LossGrad(
    device const T* input,
    device const T* target,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  const T val =  input[index] - target[index];
  const T abs_val = abs(val);
  dx[index] = abs_val < T(float_arg1) ? val / T(float_arg1)
                                      : (val > T(0)) - (val < T(0));
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(SmoothL1Loss, float);
INSTANTIATE_KERNEL(SmoothL1LossGrad, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(SmoothL1Loss, double);
INSTANTIATE_KERNEL(SmoothL1LossGrad, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                  \
  template <>                                                            \
  void name<T, MPSContext>(                                              \
      const int N,                                                       \
      const float beta,                                                  \
      const T* input,                                                    \
      const T* target,                                                   \
      T* loss,                                                           \
      MPSContext* ctx) {                                                 \
    auto kernel = MPSKernel::TypedString<T>(#name);                      \
    vector<MPSConstant> args({MPSConstant(&beta, MTLDataTypeFloat, 0)}); \
    auto* command_buffer = ctx->mps_stream()->command_buffer();          \
    auto* encoder = [command_buffer computeCommandEncoder];              \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);    \
    [encoder setComputePipelineState:pso];                               \
    [encoder setBuffer:id<MTLBuffer>(input) offset:0 atIndex:0];         \
    [encoder setBuffer:id<MTLBuffer>(target) offset:0 atIndex:1];        \
    [encoder setBuffer:id<MTLBuffer>(loss) offset:0 atIndex:2];          \
    MPSDispatchThreads(N, encoder, pso);                                 \
    [encoder endEncoding];                                               \
    [encoder release];                                                   \
  }

DEFINE_KERNEL_LAUNCHER(SmoothL1Loss, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1Loss, double);
DEFINE_KERNEL_LAUNCHER(SmoothL1LossGrad, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1LossGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
