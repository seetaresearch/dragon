#include "dragon/kernels/loss/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // C
constant uint uint_arg2 [[function_constant(1)]]; // S
constant int int_arg1 [[function_constant(2)]];   // ignore_index

template <typename T>
kernel void CrossEntropy(
    device const T* input,
    device const T* target,
    device T* loss,
    const uint index [[thread_position_in_grid]]) {
  loss[index] = -target[index] * log(max(input[index], T(FLT_MIN)));
}

template <typename T>
kernel void SigmoidCrossEntropy(
    device const T* input,
    device const T* target,
    device T* loss,
    const uint index [[thread_position_in_grid]]) {
  if (target[index] < T(0)) {
    loss[index] =  T(0);
  } else {
    const float lgt = input[index];
    loss[index] = log(1.f + exp(lgt - 2.f * lgt * (lgt >= 0.f))) +
                  lgt * ((lgt >= 0.f) - float(target[index]));
  }
}

template <typename T>
kernel void SigmoidCrossEntropyGrad(
    device const T* input,
    device const T* target,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  if (target[index] < T(0)) {
    dx[index] = T(0);
  } else {
    dx[index] = T(1) / (T(1) + exp(-input[index])) - target[index];
  }
}

template <typename InputT, typename TargetT>
kernel void SparseCrossEntropy(
    device const InputT* input,
    device const TargetT* target,
    device InputT* loss,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index / uint_arg2, j = index % uint_arg2;
  const int tgt = int(target[index]);
  if (tgt == int_arg1) {
    loss[index] = InputT(0);
  } else {
    loss[index] = -log(max(input[(i * uint_arg1 + tgt) * uint_arg2 + j],
                           InputT(FLT_MIN)));
  }
}

template <typename InputT, typename TargetT>
kernel void SparseSoftmaxCrossEntropyGrad(
    device const InputT* input, /* not used */
    device const TargetT* target,
    device InputT* dx,
    const uint index [[thread_position_in_grid]]) {
  const uint i = index / uint_arg2, j = index % uint_arg2;
  const int tgt = int(target[index]);
  if (tgt == int_arg1) {
    device InputT* offset_dx = dx + i * uint_arg1 * uint_arg2 + j;
    for (uint _ = 0; _ < uint_arg1; ++_, offset_dx += uint_arg2) {
      offset_dx[0] = InputT(0);
    }
  } else {
    dx[(i * uint_arg1 + tgt) * uint_arg2 + j] -= InputT(1);
  }
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, device T*, uint);

INSTANTIATE_KERNEL(CrossEntropy, float);
INSTANTIATE_KERNEL(SigmoidCrossEntropy, float);
INSTANTIATE_KERNEL(SigmoidCrossEntropyGrad, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(CrossEntropy, double);
INSTANTIATE_KERNEL(SigmoidCrossEntropy, double);
INSTANTIATE_KERNEL(SigmoidCrossEntropyGrad, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_KERNEL(name, InputT, TargetT) \
  template [[host_name(#name"_"#InputT"_"#TargetT)]] \
  kernel void name(device const InputT*, device const TargetT*, \
                   device InputT*, uint);

INSTANTIATE_KERNEL(SparseCrossEntropy, float, int);
INSTANTIATE_KERNEL(SparseCrossEntropy, float, int64_t);
INSTANTIATE_KERNEL(SparseSoftmaxCrossEntropyGrad, float, int);
INSTANTIATE_KERNEL(SparseSoftmaxCrossEntropyGrad, float, int64_t);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(SparseCrossEntropy, double, int);
INSTANTIATE_KERNEL(SparseCrossEntropy, double, int64_t);
INSTANTIATE_KERNEL(SparseSoftmaxCrossEntropyGrad, double, int);
INSTANTIATE_KERNEL(SparseSoftmaxCrossEntropyGrad, double, int64_t);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                           \
  template <>                                                     \
  void name<T, MPSContext>(                                       \
      const int N,                                                \
      const T* input,                                             \
      const T* target,                                            \
      T* loss,                                                    \
      MPSContext* ctx) {                                          \
    auto kernel = MPSKernel::TypedString<T>(#name);               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();   \
    auto* encoder = [command_buffer computeCommandEncoder];       \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx);   \
    [encoder setComputePipelineState:pso];                        \
    [encoder setBuffer:id<MTLBuffer>(input) offset:0 atIndex:0];  \
    [encoder setBuffer:id<MTLBuffer>(target) offset:0 atIndex:1]; \
    [encoder setBuffer:id<MTLBuffer>(loss) offset:0 atIndex:2];   \
    MPSDispatchThreads(N, encoder, pso);                          \
    [encoder endEncoding];                                        \
    [encoder release];                                            \
  }

DEFINE_KERNEL_LAUNCHER(CrossEntropy, float);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)                           \
  template <>                                                     \
  void name<T, MPSContext>(                                       \
      const int N,                                                \
      const T* input,                                             \
      const T* target,                                            \
      T* loss,                                                    \
      T* mask,                                                    \
      MPSContext* ctx) {                                          \
    auto kernel = MPSKernel::TypedString<T>(#name);               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();   \
    auto* encoder = [command_buffer computeCommandEncoder];       \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx);   \
    [encoder setComputePipelineState:pso];                        \
    [encoder setBuffer:id<MTLBuffer>(input) offset:0 atIndex:0];  \
    [encoder setBuffer:id<MTLBuffer>(target) offset:0 atIndex:1]; \
    [encoder setBuffer:id<MTLBuffer>(loss) offset:0 atIndex:2];   \
    MPSDispatchThreads(N, encoder, pso);                          \
    [encoder endEncoding];                                        \
    [encoder release];                                            \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, double);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT)                 \
  template <>                                                         \
  void name<InputT, TargetT, MPSContext>(                             \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const int ignore_index,                                         \
      const InputT* input,                                            \
      const TargetT* target,                                          \
      InputT* loss,                                                   \
      InputT* mask,                                                   \
      MPSContext* ctx) {                                              \
    const uint arg1 = C, arg2 = S;                                    \
    auto kernel = MPSKernel::TypedString<InputT>("Sparse" #name);     \
    kernel = MPSKernel::TypedString<TargetT>(kernel);                 \
    auto args = vector<MPSConstant>(                                  \
        {MPSConstant(&arg1, MTLDataTypeUInt, 0),                      \
         MPSConstant(&arg2, MTLDataTypeUInt, 1),                      \
         MPSConstant(&ignore_index, MTLDataTypeInt, 2)});             \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(input) offset:0 atIndex:0];      \
    [encoder setBuffer:id<MTLBuffer>(target) offset:0 atIndex:1];     \
    [encoder setBuffer:id<MTLBuffer>(loss) offset:0 atIndex:2];       \
    MPSDispatchThreads((N * S), encoder, pso);                        \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(CrossEntropy, float, int);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, float, int64_t);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double, int);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, float, int);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, double, int);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
