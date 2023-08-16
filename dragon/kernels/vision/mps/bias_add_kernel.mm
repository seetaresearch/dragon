#include "dragon/kernels/vision/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // C
constant uint uint_arg2 [[function_constant(1)]]; // S

template <typename T>
kernel void BiasAdd(
    device const T* x,
    device const T* bias,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index] + bias[index % uint_arg1];
}

template <typename T>
kernel void SpatialBiasAdd(
    device const T* x,
    device const T* bias,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index] + bias[index / uint_arg2 % uint_arg1];
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(BiasAdd, half);
INSTANTIATE_KERNEL(BiasAdd, float);
INSTANTIATE_KERNEL(SpatialBiasAdd, half);
INSTANTIATE_KERNEL(SpatialBiasAdd, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(BiasAdd, double);
INSTANTIATE_KERNEL(SpatialBiasAdd, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void BiasAdd<T, MPSContext>(                                                \
      const int N,                                                            \
      const int S,                                                            \
      const int C,                                                            \
      const T* x,                                                             \
      const T* bias,                                                          \
      T* y,                                                                   \
      MPSContext* ctx) {                                                      \
    const uint arg1 = C, arg2 = S;                                            \
    auto kernel = MPSKernel::TypedString<T>("BiasAdd");                       \
    vector<MPSConstant> args({MPSConstant(&arg1, MTLDataTypeUInt, 0)});       \
    MTLComputePipelineState_t pso = nil;                                      \
    if (S == 1) {                                                             \
      pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);             \
    } else {                                                                  \
      args.emplace_back(MPSConstant(&arg2, MTLDataTypeUInt, 1));              \
      pso = MPSKernel("Spatial" + kernel, METAL_SHADERS).GetState(ctx, args); \
    }                                                                         \
    auto* command_buffer = ctx->mps_stream()->command_buffer();               \
    auto* encoder = [command_buffer computeCommandEncoder];                   \
    [encoder setComputePipelineState:pso];                                    \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                  \
    [encoder setBuffer:id<MTLBuffer>(bias) offset:0 atIndex:1];               \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:2];                  \
    MPSDispatchThreads((N * C * S), encoder, pso);                            \
    [encoder endEncoding];                                                    \
    [encoder release];                                                        \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
