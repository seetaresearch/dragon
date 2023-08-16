#include "dragon/kernels/math/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void CosGrad(
    device const T* dy,
    device const T* x,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  dx[index] = -dy[index] * sin(x[index]);
}

template <typename T>
kernel void ReciprocalGrad(
    device const T* dy,
    device const T* y,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = y[index];
  dx[index] = -dy[index] * val * val;
}

template <typename T>
kernel void RsqrtGrad(
    device const T* dy,
    device const T* y,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  const T val = y[index];
  dx[index] = T(-0.5) * dy[index] * val * val * val;
}

template <typename T>
kernel void SinGrad(
    device const T* dy,
    device const T* x,
    device T* dx, 
    const uint index [[thread_position_in_grid]]) {
  dx[index] = dy[index] * cos(x[index]);
}

#define INSTANTIATE_GRAD_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_GRAD_KERNEL(CosGrad, half);
INSTANTIATE_GRAD_KERNEL(CosGrad, float);
INSTANTIATE_GRAD_KERNEL(ReciprocalGrad, half);
INSTANTIATE_GRAD_KERNEL(ReciprocalGrad, float);
INSTANTIATE_GRAD_KERNEL(RsqrtGrad, half);
INSTANTIATE_GRAD_KERNEL(RsqrtGrad, float);
INSTANTIATE_GRAD_KERNEL(SinGrad, half);
INSTANTIATE_GRAD_KERNEL(SinGrad, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(CosGrad, double);
INSTANTIATE_GRAD_KERNEL(ReciprocalGrad, double);
INSTANTIATE_GRAD_KERNEL(RsqrtGrad, double);
INSTANTIATE_GRAD_KERNEL(SinGrad, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

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

DEFINE_GRAD_KERNEL_LAUNCHER(CosGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(CosGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(CosGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(CosGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ReciprocalGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReciprocalGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(ReciprocalGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ReciprocalGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(RsqrtGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(RsqrtGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(RsqrtGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(RsqrtGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SinGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SinGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(SinGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SinGrad, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
