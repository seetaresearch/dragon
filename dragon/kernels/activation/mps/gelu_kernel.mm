#include "dragon/kernels/activation/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void ApproxGelu(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const float val = float(x[index]);
  y[index] = T(fma(
      val,
      tanh(float(0.7978845608028654) * fma(float(0.044715), val * val * val, val)),
      val) * 0.5f);
}

template <typename T>
kernel void ApproxGeluGrad(
    device const T* dy,
    device const T* x,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const float val = float(x[index]);
  const float val2 = tanh(float(0.7978845608028654) *
                          fma(float(0.044715), val * val * val, val));
  dx[index] = T(float(dy[index]) * 0.5f * fma(
    fma(-val, val2 * val2, val),
    fma(float(0.10703222440890037), val * val, float(0.7978845608028654)),
    val2 + 1.f));
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device T*, uint);

#define INSTANTIATE_GRAD_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(ApproxGelu, half);
INSTANTIATE_KERNEL(ApproxGelu, float);
INSTANTIATE_GRAD_KERNEL(ApproxGeluGrad, half);
INSTANTIATE_GRAD_KERNEL(ApproxGeluGrad, float);
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
      const int N, const T* dy, const T* x, T* dx, MPSContext* ctx) { \
    auto kernel = MPSKernel::TypedString<T>(#name);                   \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx);       \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:0];         \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:1];          \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:2];         \
    MPSDispatchThreads(N, encoder, pso);                              \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(Gelu, float16);
DEFINE_KERNEL_LAUNCHER(Gelu, float);
DEFINE_KERNEL_LAUNCHER(Gelu, double);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float16);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, double);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
