#include "dragon/kernels/math/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]];
constant float float_arg2 [[function_constant(1)]];

template <typename T>
kernel void Clip(
    device const T* x,
    device T* y, 
    const uint index [[thread_position_in_grid]]) {
  y[index] = clamp(x[index], T(float_arg1), T(float_arg2));
}

template <typename T>
kernel void ClipGrad(
    device const T* dy,
    device const T* x,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  const T val = x[index];
  dx[index] = (val < T(float_arg1) || val > T(float_arg2)) ? T(0) : dy[index];
}

#define INSTANTIATE_KERNEL(T) \
  template [[host_name("Clip_"#T)]] \
  kernel void Clip(device const T*, device T*, uint);

#define INSTANTIATE_GRAD_KERNEL(T) \
  template [[host_name("ClipGrad_"#T)]] \
  kernel void ClipGrad(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(uint8_t);
INSTANTIATE_KERNEL(int8_t);
INSTANTIATE_KERNEL(int);
INSTANTIATE_KERNEL(int64_t);
INSTANTIATE_KERNEL(half);
INSTANTIATE_KERNEL(float);
INSTANTIATE_GRAD_KERNEL(half);
INSTANTIATE_GRAD_KERNEL(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(double);
INSTANTIATE_GRAD_KERNEL(double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void Clip<T, MPSContext>(                                           \
      const int N,                                                    \
      const float low,                                                \
      const float high,                                               \
      const T* x,                                                     \
      T* y,                                                           \
      MPSContext* ctx) {                                              \
    auto kernel = MPSKernel::TypedString<T>("Clip");                  \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&low, MTLDataTypeFloat, 0),                       \
        MPSConstant(&high, MTLDataTypeFloat, 1),                      \
    });                                                               \
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

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void ClipGrad<T, MPSContext>(                                       \
      const int N,                                                    \
      const float low,                                                \
      const float high,                                               \
      const T* dy,                                                    \
      const T* x,                                                     \
      T* dx,                                                          \
      MPSContext* ctx) {                                              \
    auto kernel = MPSKernel::TypedString<T>("ClipGrad");              \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&low, MTLDataTypeFloat, 0),                       \
        MPSConstant(&high, MTLDataTypeFloat, 1),                      \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:0];         \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:1];          \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:2];         \
    MPSDispatchThreads(N, encoder, pso);                              \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
