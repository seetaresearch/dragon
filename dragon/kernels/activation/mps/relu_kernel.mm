#include "dragon/kernels/activation/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]];

template <typename T>
kernel void Relu(
    device const T* x,
    device T* y, 
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  y[index] = val > T(0) ? val : val * T(float_arg1);
}

template <typename T>
kernel void ReluN(
    device const T* x,
    device T* y, 
    const uint index [[thread_position_in_grid]]) {
  y[index] = clamp(x[index], T(0), T(float_arg1));
}

template <typename T>
kernel void ReluGrad(
    device const T* dy,
    device const T* y,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = y[index];
  dx[index] = dy[index] * ((val > T(0)) + T(float_arg1) * (val <= T(0)));
}

template <typename T>
kernel void ReluNGrad(
    device const T* dy,
    device const T* y,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = y[index];
  dx[index] = ((val > T(0)) && val < T(float_arg1)) ? dy[index] : T(0);
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device T*, uint);

#define INSTANTIATE_GRAD_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(Relu, half);
INSTANTIATE_KERNEL(Relu, float);
INSTANTIATE_KERNEL(ReluN, half);
INSTANTIATE_KERNEL(ReluN, float);
INSTANTIATE_GRAD_KERNEL(ReluGrad, half);
INSTANTIATE_GRAD_KERNEL(ReluGrad, float);
INSTANTIATE_GRAD_KERNEL(ReluNGrad, half);
INSTANTIATE_GRAD_KERNEL(ReluNGrad, float);
#undef INSTANTIATE_KERNEL
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                    \
  template <>                                                              \
  void name<T, MPSContext>(                                                \
      const int N, const float alpha, const T* x, T* y, MPSContext* ctx) { \
    auto kernel = MPSKernel::TypedString<T>(#name);                        \
    vector<MPSConstant> args({MPSConstant(&alpha, MTLDataTypeFloat, 0)});  \
    auto* command_buffer = ctx->mps_stream()->command_buffer();            \
    auto* encoder = [command_buffer computeCommandEncoder];                \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);      \
    [encoder setComputePipelineState:pso];                                 \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];               \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];               \
    MPSDispatchThreads(N, encoder, pso);                                   \
    [encoder endEncoding];                                                 \
    [encoder release];                                                     \
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

DEFINE_KERNEL_LAUNCHER(Relu, float16);
DEFINE_KERNEL_LAUNCHER(Relu, float);
DEFINE_KERNEL_LAUNCHER(Relu, double);
DEFINE_KERNEL_LAUNCHER(ReluN, float16);
DEFINE_KERNEL_LAUNCHER(ReluN, float);
DEFINE_KERNEL_LAUNCHER(ReluN, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ReluNGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
