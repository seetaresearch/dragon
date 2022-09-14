#include "dragon/kernels/activation/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void Silu(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  y[index] = val / (T(1) + exp(-val));
}

template <typename T>
kernel void HardSwish(
    device const T* x,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  const T s_val = fma(val, T(0.1666666666666667), T(0.5));
  y[index] = val * max(T(0), min(T(1), s_val));
}

template <typename T>
kernel void SiluGrad(
    device const T* dy,
    device const T* x,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  const T s_val = T(1) / (T(1) + exp(-val));
  dx[index] = dy[index] * s_val * (val + T(1) - val * s_val);
}

template <typename T>
kernel void HardSwishGrad(
    device const T* dy,
    device const T* x,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  dx[index] = (val < T(-3)) ? T(0) : (val < T(3)) ?
    dy[index] * fma(val, T(0.3333333333333333), T(0.5)) : dy[index];
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device T*, uint);

#define INSTANTIATE_GRAD_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(Silu, half);
INSTANTIATE_KERNEL(Silu, float);
INSTANTIATE_KERNEL(HardSwish, half);
INSTANTIATE_KERNEL(HardSwish, float);
INSTANTIATE_GRAD_KERNEL(SiluGrad, half);
INSTANTIATE_GRAD_KERNEL(SiluGrad, float);
INSTANTIATE_GRAD_KERNEL(HardSwishGrad, half);
INSTANTIATE_GRAD_KERNEL(HardSwishGrad, float);
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

DEFINE_KERNEL_LAUNCHER(Silu, float16);
DEFINE_KERNEL_LAUNCHER(Silu, float);
DEFINE_KERNEL_LAUNCHER(Silu, double);
DEFINE_KERNEL_LAUNCHER(HardSwish, float16);
DEFINE_KERNEL_LAUNCHER(HardSwish, float);
DEFINE_KERNEL_LAUNCHER(HardSwish, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
