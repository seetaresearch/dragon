#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]]; // start
constant float float_arg2 [[function_constant(1)]]; // delta
constant uint uint_arg1 [[function_constant(2)]];   // M
constant uint uint_arg2 [[function_constant(3)]];   // N
constant int int_arg1 [[function_constant(4)]];     // k

template <typename T>
kernel void Range(
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = T(float_arg1 + float(index) * float_arg2);
}

template <typename T>
kernel void SetEyeUpper(
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index % uint_arg1;
  const uint j = i + uint(int_arg1);
  y[index * uint_arg2 + min(j, uint_arg2 - 1)] = j < uint_arg2 ? T(1) : T(0);
}

template <typename T>
kernel void SetEyeLower(
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index % uint_arg1;
  const int j = int(i) - int_arg1;
  y[index * uint_arg2 + max(j, 0)] = j < 0 ? T(0) : T(1);
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device T*, uint);

INSTANTIATE_KERNEL(Range, uint8_t);
INSTANTIATE_KERNEL(Range, int8_t);
INSTANTIATE_KERNEL(Range, int);
INSTANTIATE_KERNEL(Range, int64_t);
INSTANTIATE_KERNEL(Range, half);
INSTANTIATE_KERNEL(Range, float);
INSTANTIATE_KERNEL(SetEyeUpper, bool);
INSTANTIATE_KERNEL(SetEyeUpper, uint8_t);
INSTANTIATE_KERNEL(SetEyeUpper, int8_t);
INSTANTIATE_KERNEL(SetEyeUpper, int);
INSTANTIATE_KERNEL(SetEyeUpper, int64_t);
INSTANTIATE_KERNEL(SetEyeUpper, half);
INSTANTIATE_KERNEL(SetEyeUpper, float);
INSTANTIATE_KERNEL(SetEyeLower, bool);
INSTANTIATE_KERNEL(SetEyeLower, uint8_t);
INSTANTIATE_KERNEL(SetEyeLower, int8_t);
INSTANTIATE_KERNEL(SetEyeLower, int);
INSTANTIATE_KERNEL(SetEyeLower, int64_t);
INSTANTIATE_KERNEL(SetEyeLower, half);
INSTANTIATE_KERNEL(SetEyeLower, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(Range, double);
INSTANTIATE_KERNEL(SetEyeUpper, double);
INSTANTIATE_KERNEL(SetEyeLower, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void Range<T, MPSContext>(                                          \
      const int N,                                                    \
      const double start,                                             \
      const double delta,                                             \
      T* y,                                                           \
      MPSContext* ctx) {                                              \
    const float arg1 = start, arg2 = delta;                           \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&arg1, MTLDataTypeFloat, 0),                      \
        MPSConstant(&arg2, MTLDataTypeFloat, 1),                      \
    });                                                               \
    auto kernel = MPSKernel::TypedString<T>("Range");                 \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:0];          \
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
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void SetEye<T, MPSContext>(                                         \
      const int batch_size,                                           \
      const int M,                                                    \
      const int N,                                                    \
      const int k,                                                    \
      T* y,                                                           \
      MPSContext* ctx) {                                              \
    const auto nthreads = batch_size * M;                             \
    math::Set((nthreads * N), convert::To<T>(0.f), y, ctx);           \
    const uint arg1 = M, arg2 = N;                                    \
    const int arg3 = k > 0 ? k : -k;                                  \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&arg1, MTLDataTypeUInt, 2),                       \
        MPSConstant(&arg2, MTLDataTypeUInt, 3),                       \
        MPSConstant(&arg3, MTLDataTypeInt, 4),                        \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto kernel = string("SetEye") + (k > 0 ? "Upper" : "Lower");     \
    kernel = MPSKernel::TypedString<T>(kernel);                       \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:0];          \
    MPSDispatchThreads(nthreads, encoder, pso);                       \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
