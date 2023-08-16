#include "dragon/kernels/training/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]]; //lr
constant float float_arg2 [[function_constant(1)]]; // beta1
constant float float_arg3 [[function_constant(2)]]; // beta2
constant float float_arg4 [[function_constant(3)]]; // eps
constant float float_arg5 [[function_constant(4)]]; // wd

template <typename T>
kernel void Adam(
    device const T* x,
    device const T* g,
    device T* m,
    device T* v,
    device T* y,
    const uint i [[thread_position_in_grid]]) {
  const T gi = float_arg5 > 0.f ? fma(T(float_arg5), x[i], g[i]) : g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], T(1.f - float_arg2) * gi);
  const T vi = v[i] = fma(T(float_arg3), v[i], T(1.f - float_arg3) * gi * gi);
  y[i] -= T(float_arg1) * mi / (sqrt(vi) + T(float_arg4));
}

template <typename T, typename CopyT>
kernel void Adam(
    device const T* x,
    device const T* g,
    device T* m,
    device T* v,
    device T* y,
    device CopyT* y_copy,
    const uint i [[thread_position_in_grid]]) {
  const T gi = float_arg5 > 0.f ? fma(T(float_arg5), x[i], g[i]) : g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], T(1.f - float_arg2) * gi);
  const T vi = v[i] = fma(T(float_arg3), v[i], T(1.f - float_arg3) * gi * gi);
  y[i] -= T(float_arg1) * mi / (sqrt(vi) + T(float_arg4));
  y_copy[i] = CopyT(y[i]);
}

template <typename T>
kernel void AdamW(
    device const T* x,
    device const T* g,
    device T* m,
    device T* v,
    device T* y,
    const uint i [[thread_position_in_grid]]) {
  const T gi = g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], T(1.f - float_arg2) * gi);
  const T vi = v[i] = fma(T(float_arg3), v[i], T(1.f - float_arg3) * gi * gi);
  const T ui = T(float_arg1) * mi / (sqrt(vi) + T(float_arg4));
  y[i] -= float_arg5 > 0.f ? fma(T(float_arg5), x[i], ui) : ui;
}

template <typename T, typename CopyT>
kernel void AdamW(
    device const T* x,
    device const T* g,
    device T* m,
    device T* v,
    device T* y,
    device CopyT* y_copy,
    const uint i [[thread_position_in_grid]]) {
  const T gi = g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], T(1.f - float_arg2) * gi);
  const T vi = v[i] = fma(T(float_arg3), v[i], T(1.f - float_arg3) * gi * gi);
  const T ui = T(float_arg1) * mi / (sqrt(vi) + T(float_arg4));
  y[i] -= float_arg5 > 0.f ? fma(T(float_arg5), x[i], ui) : ui;
  y_copy[i] = CopyT(y[i]);
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, \
                   device T*, device T*, device T*, uint);
INSTANTIATE_KERNEL(Adam, float);
INSTANTIATE_KERNEL(AdamW, float);
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_KERNEL(name, T, CopyT) \
  template [[host_name(#name"WithCopy_"#T)]] \
  kernel void name(device const T*, device const T*, \
                   device T*, device T*, device T*, device CopyT*, uint);
INSTANTIATE_KERNEL(Adam, float, half);
INSTANTIATE_KERNEL(AdamW, float, half);
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT)                                \
  template <>                                                                 \
  void name<T, CopyT, MPSContext>(                                            \
      const int N,                                                            \
      const float lr,                                                         \
      const float beta1,                                                      \
      const float beta2,                                                      \
      const float eps,                                                        \
      const float wd,                                                         \
      const T* x,                                                             \
      const T* g,                                                             \
      T* m,                                                                   \
      T* v,                                                                   \
      T* y,                                                                   \
      CopyT* y_copy,                                                          \
      MPSContext* ctx) {                                                      \
    auto kernel = string(#name) + (y_copy ? "WithCopy" : "");                 \
    kernel = MPSKernel::TypedString<T>(kernel);                               \
    auto args = vector<MPSConstant>({                                         \
        MPSConstant(&lr, MTLDataTypeFloat, 0),                                \
        MPSConstant(&beta1, MTLDataTypeFloat, 1),                             \
        MPSConstant(&beta2, MTLDataTypeFloat, 2),                             \
        MPSConstant(&eps, MTLDataTypeFloat, 3),                               \
        MPSConstant(&wd, MTLDataTypeFloat, 4),                                \
    });                                                                       \
    auto* command_buffer = ctx->mps_stream()->command_buffer();               \
    auto* encoder = [command_buffer computeCommandEncoder];                   \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);         \
    [encoder setComputePipelineState:pso];                                    \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                  \
    [encoder setBuffer:id<MTLBuffer>(g) offset:0 atIndex:1];                  \
    [encoder setBuffer:id<MTLBuffer>(m) offset:0 atIndex:2];                  \
    [encoder setBuffer:id<MTLBuffer>(v) offset:0 atIndex:3];                  \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:4];                  \
    if (y_copy) [encoder setBuffer:id<MTLBuffer>(y_copy) offset:0 atIndex:5]; \
    MPSDispatchThreads(N, encoder, pso);                                      \
    [encoder endEncoding];                                                    \
    [encoder release];                                                        \
  }

DEFINE_KERNEL_LAUNCHER(Adam, float, float16);
DEFINE_KERNEL_LAUNCHER(Adam, float, bfloat16);
DEFINE_KERNEL_LAUNCHER(Adam, float, float);
DEFINE_KERNEL_LAUNCHER(Adam, double, double);
DEFINE_KERNEL_LAUNCHER(AdamW, float, float16);
DEFINE_KERNEL_LAUNCHER(AdamW, float, bfloat16);
DEFINE_KERNEL_LAUNCHER(AdamW, float, float);
DEFINE_KERNEL_LAUNCHER(AdamW, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
