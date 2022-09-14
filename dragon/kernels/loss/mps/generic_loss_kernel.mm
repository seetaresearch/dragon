#include "dragon/kernels/loss/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // CxS
constant uint uint_arg2 [[function_constant(1)]]; // S
constant float float_arg1 [[function_constant(2)]]; // scale

template <typename T>
kernel void ReduceLossGrad(
    device const T* dl, // []
    device T* dx,       // [N]
    const uint index [[thread_position_in_grid]]) {
  dx[index] *= dl[0] * T(float_arg1);
}

template <typename T>
kernel void BroadcastLossGrad(
    device const T* dl, // [NxS]
    device T* dx,       // [NxCxS]
    const uint index [[thread_position_in_grid]]) {
  dx[index] *= dl[index / uint_arg1 * uint_arg2 + index % uint_arg2];
}

#define INSTANTIATE_GRAD_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device T*, uint);

INSTANTIATE_GRAD_KERNEL(ReduceLossGrad, half);
INSTANTIATE_GRAD_KERNEL(ReduceLossGrad, float);
INSTANTIATE_GRAD_KERNEL(BroadcastLossGrad, half);
INSTANTIATE_GRAD_KERNEL(BroadcastLossGrad, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(ReduceLossGrad, double);
INSTANTIATE_GRAD_KERNEL(BroadcastLossGrad, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                       \
  template <>                                                           \
  void ReduceLoss<T, MPSContext>(                                       \
      const int N,                                                      \
      const int num_masks,                                              \
      const float normalizer,                                           \
      const T* x,                                                       \
      const T* mask,                                                    \
      T* y,                                                             \
      MPSContext* ctx) {                                                \
    if (num_masks > 0 && normalizer < 0.f) {                            \
      LOG(FATAL) << "Loss normalized by valid masks is not supported."; \
    } else {                                                            \
      math::Sum(N, 1.f / std::max(1.f, normalizer), x, y, ctx);         \
    }                                                                   \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                      \
  template <>                                                               \
  void ReduceLossGrad<T, MPSContext>(                                       \
      const int N,                                                          \
      const int num_masks,                                                  \
      const float normalizer,                                               \
      const T* dl,                                                          \
      const T* mask,                                                        \
      T* dx,                                                                \
      MPSContext* ctx) {                                                    \
    if (num_masks > 0 && normalizer < 0.f) {                                \
      LOG(FATAL) << "Loss normalized by valid masks is not supported.";     \
    } else {                                                                \
      const float scale = 1.f / std::max(0.5f, normalizer);                 \
      auto kernel = MPSKernel::TypedString<T>("ReduceLossGrad");            \
      vector<MPSConstant> args({MPSConstant(&scale, MTLDataTypeFloat, 2)}); \
      auto* command_buffer = ctx->mps_stream()->command_buffer();           \
      auto* encoder = [command_buffer computeCommandEncoder];               \
      auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);     \
      [encoder setComputePipelineState:pso];                                \
      [encoder setBuffer:id<MTLBuffer>(dl) offset:0 atIndex:0];             \
      [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:1];             \
      MPSDispatchThreads(N, encoder, pso);                                  \
      [encoder endEncoding];                                                \
      [encoder release];                                                    \
    }                                                                       \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void BroadcastLossGrad<T, MPSContext>(                              \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const T* dl,                                                    \
      T* dx,                                                          \
      MPSContext* ctx) {                                              \
    const uint arg1 = C * S, arg2 = S;                                \
    auto kernel = MPSKernel::TypedString<T>("BroadcastLossGrad");     \
    auto args = vector<MPSConstant>(                                  \
        {MPSConstant(&arg1, MTLDataTypeUInt, 0),                      \
         MPSConstant(&arg2, MTLDataTypeUInt, 1)});                    \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(dl) offset:0 atIndex:0];         \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:1];         \
    MPSDispatchThreads((N * C * S), encoder, pso);                    \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
