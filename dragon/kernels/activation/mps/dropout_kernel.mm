#include "dragon/kernels/activation/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]]; // ratio
constant float float_arg2 [[function_constant(1)]]; // scale
constant uint uint_arg1 [[function_constant(2)]];   // C

template <typename T>
kernel void Dropout(
    device const float* r,
    device const T* x,
    device T* y,
    device uint8_t* mask,
    const uint index [[thread_position_in_grid]]) {
  const T alpha = float(mask[index] = (r[index] > float_arg1)) * float_arg2;
  y[index] = x[index] * alpha;
}

template <typename T>
kernel void DropPath(
    device const float* r,
    device const T* x,
    device T* y,
    device uint8_t* mask,
    const uint index [[thread_position_in_grid]]) {
  const uint j = index / uint_arg1;
  const T alpha = float(mask[j] = (r[j] > float_arg1)) * float_arg2;
  y[index] = x[index] * alpha;
}

template <typename T>
kernel void DropPathGrad(
    device const uint8_t* mask,
    device const T* dy,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  const T alpha = float(mask[index / uint_arg1]) * float_arg2;
  dx[index] = dy[index] * alpha;
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const float*, device const T*, \
                   device T*, device uint8_t*, uint);

INSTANTIATE_KERNEL(Dropout, half);
INSTANTIATE_KERNEL(Dropout, float);
INSTANTIATE_KERNEL(DropPath, half);
INSTANTIATE_KERNEL(DropPath, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(Dropout, double);
INSTANTIATE_KERNEL(DropPath, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const uint8_t*, device const T*, device T*, uint);

INSTANTIATE_GRAD_KERNEL(DropPathGrad, half);
INSTANTIATE_GRAD_KERNEL(DropPathGrad, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(DropPathGrad, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void Dropout<T, MPSContext>(                                        \
      const int N,                                                    \
      const float ratio,                                              \
      const float scale,                                              \
      const float* r,                                                 \
      const T* x,                                                     \
      T* y,                                                           \
      uint8_t* mask,                                                  \
      MPSContext* ctx) {                                              \
    auto kernel = MPSKernel::TypedString<T>("Dropout");               \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&ratio, MTLDataTypeFloat, 0),                     \
        MPSConstant(&scale, MTLDataTypeFloat, 1),                     \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(r) offset:0 atIndex:0];          \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:1];          \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:2];          \
    [encoder setBuffer:id<MTLBuffer>(mask) offset:0 atIndex:3];       \
    MPSDispatchThreads(N, encoder, pso);                              \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void DropPath<T, MPSContext>(                                       \
      const int N,                                                    \
      const int C,                                                    \
      const float ratio,                                              \
      const float scale,                                              \
      const float* r,                                                 \
      const T* x,                                                     \
      T* y,                                                           \
      uint8_t* mask,                                                  \
      MPSContext* ctx) {                                              \
    const uint arg3 = C;                                              \
    auto kernel = MPSKernel::TypedString<T>("DropPath");              \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&ratio, MTLDataTypeFloat, 0),                     \
        MPSConstant(&scale, MTLDataTypeFloat, 1),                     \
        MPSConstant(&arg3, MTLDataTypeUInt, 2),                       \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(r) offset:0 atIndex:0];          \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:1];          \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:2];          \
    [encoder setBuffer:id<MTLBuffer>(mask) offset:0 atIndex:3];       \
    MPSDispatchThreads((N * C), encoder, pso);                        \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void DropPathGrad<T, MPSContext>(                                   \
      const int N,                                                    \
      const int C,                                                    \
      const float scale,                                              \
      const uint8_t* mask,                                            \
      const T* dy,                                                    \
      T* dx,                                                          \
      MPSContext* ctx) {                                              \
    const uint arg2 = C;                                              \
    auto kernel = MPSKernel::TypedString<T>("DropPathGrad");          \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&scale, MTLDataTypeFloat, 1),                     \
        MPSConstant(&arg2, MTLDataTypeUInt, 2),                       \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(mask) offset:0 atIndex:0];       \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:1];         \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:2];         \
    MPSDispatchThreads((N * C), encoder, pso);                        \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
