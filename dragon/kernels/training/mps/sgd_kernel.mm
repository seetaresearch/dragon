#include "dragon/kernels/training/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant float float_arg1 [[function_constant(0)]];
constant float float_arg2 [[function_constant(1)]];
constant float float_arg3 [[function_constant(2)]];

template <typename T>
kernel void MomentumSGD(
    device const T* x,
    device const T* g,
    device T* m,
    device T* y,
    const uint i [[thread_position_in_grid]]) {
  const T gi = float_arg3 > 0.f ? fma(T(float_arg3), x[i], g[i]) : g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], gi);
  y[i] -= T(float_arg1) * mi;
}

template <typename T, typename CopyT>
kernel void MomentumSGD(
    device const T* x,
    device const T* g,
    device T* m,
    device T* y,
    device CopyT* y_copy,
    const uint i [[thread_position_in_grid]]) {
  const T gi = float_arg3 > 0.f ? fma(T(float_arg3), x[i], g[i]) : g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], gi);
  y[i] -= T(float_arg1) * mi;
  y_copy[i] = CopyT(y[i]);
}

template <typename T>
kernel void NesterovSGD(
    device const T* x,
    device const T* g,
    device T* m,
    device T* y,
    const uint i [[thread_position_in_grid]]) {
  const T gi = float_arg3 > 0.f ? fma(T(float_arg3), x[i], g[i]) : g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], gi);
  y[i] -= T(float_arg1) * fma(T(float_arg2), mi, gi);
}

template <typename T, typename CopyT>
kernel void NesterovSGD(
    device const T* x,
    device const T* g,
    device T* m,
    device T* y,
    device CopyT* y_copy,
    const uint i [[thread_position_in_grid]]) {
  const T gi = float_arg3 > 0.f ? fma(T(float_arg3), x[i], g[i]) : g[i];
  const T mi = m[i] = fma(T(float_arg2), m[i], gi);
  y[i] -= T(float_arg1) * fma(T(float_arg2), mi, gi);
  y_copy[i] = CopyT(y[i]);
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, \
                   device T*, device T*, uint);
INSTANTIATE_KERNEL(MomentumSGD, float);
INSTANTIATE_KERNEL(NesterovSGD, float);
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_KERNEL(name, T, CopyT) \
  template [[host_name(#name"WithCopy_"#T)]] \
  kernel void name(device const T*, device const T*, \
                   device T*, device T*, device CopyT*, uint);
INSTANTIATE_KERNEL(MomentumSGD, float, half);
INSTANTIATE_KERNEL(NesterovSGD, float, half);
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT)                                \
  template <>                                                                 \
  void name<T, CopyT, MPSContext>(                                            \
      const int N,                                                            \
      const float lr,                                                         \
      const float momentum,                                                   \
      const float wd,                                                         \
      const T* x,                                                             \
      const T* g,                                                             \
      T* m,                                                                   \
      T* y,                                                                   \
      CopyT* y_copy,                                                          \
      MPSContext* ctx) {                                                      \
    auto kernel = string(#name) + (y_copy ? "WithCopy" : "");                 \
    kernel = MPSKernel::TypedString<T>(kernel);                               \
    auto args = vector<MPSConstant>({                                         \
        MPSConstant(&lr, MTLDataTypeFloat, 0),                                \
        MPSConstant(&momentum, MTLDataTypeFloat, 1),                          \
        MPSConstant(&wd, MTLDataTypeFloat, 2),                                \
    });                                                                       \
    auto* command_buffer = ctx->mps_stream()->command_buffer();               \
    auto* encoder = [command_buffer computeCommandEncoder];                   \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);         \
    [encoder setComputePipelineState:pso];                                    \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                  \
    [encoder setBuffer:id<MTLBuffer>(g) offset:0 atIndex:1];                  \
    [encoder setBuffer:id<MTLBuffer>(m) offset:0 atIndex:2];                  \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:3];                  \
    if (y_copy) [encoder setBuffer:id<MTLBuffer>(y_copy) offset:0 atIndex:4]; \
    MPSDispatchThreads(N, encoder, pso);                                      \
    [encoder endEncoding];                                                    \
    [encoder release];                                                        \
  }

DEFINE_KERNEL_LAUNCHER(MomentumSGD, float, float16);
DEFINE_KERNEL_LAUNCHER(MomentumSGD, float, float);
DEFINE_KERNEL_LAUNCHER(MomentumSGD, double, double);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, float, float16);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, float, float);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
