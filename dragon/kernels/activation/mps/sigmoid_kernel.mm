#include "dragon/kernels/activation/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]];
constant float float_arg2 [[function_constant(1)]];

template <typename T>
kernel void Sigmoid(
    device const T* x,
    device T* y, 
    const uint index [[thread_position_in_grid]]) {
  y[index] = T(1) / (T(1) + exp(-x[index]));
}

template <typename T>
kernel void HardSigmoid(
    device const T* x,
    device T* y, 
    const uint index [[thread_position_in_grid]]) {
  y[index] = clamp(fma(x[index], T(float_arg1), T(float_arg2)), T(0), T(1));
}

template <typename T>
kernel void SigmoidGrad(
    device const T* dy,
    device const T* y,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = y[index];
  dx[index] = dy[index] * val * (T(1) - val);
}

template <typename T>
kernel void HardSigmoidGrad(
    device const T* dy,
    device const T* y,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = y[index];
  dx[index] = (val > T(0) && val < T(1)) ? dy[index] * T(float_arg1) : T(0);
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device T*, uint);

#define INSTANTIATE_GRAD_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(Sigmoid, half);
INSTANTIATE_KERNEL(Sigmoid, float);
INSTANTIATE_KERNEL(HardSigmoid, half);
INSTANTIATE_KERNEL(HardSigmoid, float);
INSTANTIATE_GRAD_KERNEL(SigmoidGrad, half);
INSTANTIATE_GRAD_KERNEL(SigmoidGrad, float);
INSTANTIATE_GRAD_KERNEL(HardSigmoidGrad, half);
INSTANTIATE_GRAD_KERNEL(HardSigmoidGrad, float);
#undef INSTANTIATE_KERNEL
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                      \
  template <>                                                                \
  void name<T, MPSContext>(const int N, const T* x, T* y, MPSContext* ctx) { \
    auto kernel = MPSKernel::TypedString<T>(#name);                          \
    auto* command_buffer = ctx->mps_stream()->command_buffer();              \
    auto* encoder = [command_buffer computeCommandEncoder];                  \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx);              \
    [encoder setComputePipelineState:pso];                                   \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                 \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];                 \
    MPSDispatchThreads(N, encoder, pso);                                     \
    [encoder endEncoding];                                                   \
    [encoder release];                                                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                         \
  void name<T, MPSContext>(                                           \
      const int N, const T* dy, const T* y, T* dx, MPSContext* ctx) { \
    auto kernel = MPSKernel::TypedString<T>(#name);                   \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx);       \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:0];         \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];          \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:2];         \
    MPSDispatchThreads(N, encoder, pso);                              \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(Sigmoid, float16);
DEFINE_KERNEL_LAUNCHER(Sigmoid, bfloat16);
DEFINE_KERNEL_LAUNCHER(Sigmoid, float);
DEFINE_KERNEL_LAUNCHER(Sigmoid, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SigmoidGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SigmoidGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(SigmoidGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SigmoidGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void name<T, MPSContext>(                                           \
      const int N,                                                    \
      const float alpha,                                              \
      const float beta,                                               \
      const T* x,                                                     \
      T* y,                                                           \
      MPSContext* ctx) {                                              \
    auto kernel = MPSKernel::TypedString<T>(#name);                   \
    auto args = vector<MPSConstant>(                                  \
        {MPSConstant(&alpha, MTLDataTypeFloat, 0),                    \
         MPSConstant(&beta, MTLDataTypeFloat, 1)});                   \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];          \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];          \
    MPSDispatchThreads(N, encoder, pso);                              \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                              \
  template <>                                                             \
  void name<T, MPSContext>(                                               \
      const int N,                                                        \
      const float alpha,                                                  \
      const T* dy,                                                        \
      const T* y,                                                         \
      T* dx,                                                              \
      MPSContext* ctx) {                                                  \
    auto kernel = MPSKernel::TypedString<T>(#name);                       \
    vector<MPSConstant> args({MPSConstant(&alpha, MTLDataTypeFloat, 0)}); \
    auto* command_buffer = ctx->mps_stream()->command_buffer();           \
    auto* encoder = [command_buffer computeCommandEncoder];               \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);     \
    [encoder setComputePipelineState:pso];                                \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:0];             \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];              \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:2];             \
    MPSDispatchThreads(N, encoder, pso);                                  \
    [encoder endEncoding];                                                \
    [encoder release];                                                    \
  }

DEFINE_KERNEL_LAUNCHER(HardSigmoid, float16);
DEFINE_KERNEL_LAUNCHER(HardSigmoid, bfloat16);
DEFINE_KERNEL_LAUNCHER(HardSigmoid, float);
DEFINE_KERNEL_LAUNCHER(HardSigmoid, double);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSigmoidGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSigmoidGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSigmoidGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSigmoidGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
