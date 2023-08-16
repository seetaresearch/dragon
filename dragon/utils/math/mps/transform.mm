#include "dragon/utils/math/transform.h"
#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // C
constant uint uint_arg2 [[function_constant(1)]]; // S

template <typename T>
kernel void AffineChannel(
    device const T* x,
    device const T* scale,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = x[index] * scale[(index / uint_arg2) % uint_arg1];
}

template <typename T>
kernel void AffineChannel(
    device const T* x,
    device const T* scale,
    device const T* bias,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const uint j = (index / uint_arg2) % uint_arg1;
  y[index] = fma(x[index], scale[j], bias[j]);
}

#define INSTANTIATE_KERNEL(T) \
  template [[host_name("AffineChannel_"#T)]] \
  kernel void AffineChannel(device const T*, device const T*, device T*, uint); \
  template [[host_name("AffineChannelWithBias_"#T)]] \
  kernel void AffineChannel(device const T*, device const T*, \
                            device const T*, device T*, uint);

INSTANTIATE_KERNEL(half);
INSTANTIATE_KERNEL(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

template <typename T>
void DispatchAffine(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const bool has_bias,
    MTLComputeCommandEncoder_t encoder,
    MPSContext* ctx) {
  const auto N = math::utils::Prod(num_dims, dims);
  string kernel;
  uint arg1 = 1, arg2 = 1;
  if (num_dims == 1 && num_axes == 1 && axes[0] == 0) {
    kernel = "AffineChannel", arg1 = dims[0]; // [NxC]
  } else if (num_dims == 2 && num_axes == 1 && axes[0] == 1) {
    kernel = "AffineChannel", arg1 = dims[1]; // [N, C]
  } else if (num_dims == 2 && num_axes == 1 && axes[0] == 0) {
    kernel = "AffineChannel", arg1 = dims[0], arg2 = dims[1]; // [NxC, S]
  } else if (num_dims == 3 && num_axes == 1 && axes[0] == 1) {
    kernel = "AffineChannel", arg1 = dims[1], arg2 = dims[2]; // [N, C, S]
  } else {
    LOG(FATAL) << "Unsupported affine dimensions.";
  }
  kernel = MPSKernel::TypedString<T>(kernel + (has_bias ? "WithBias" : ""));
  auto args = vector<MPSConstant>({
      MPSConstant(&arg1, MTLDataTypeUInt, 0),
      MPSConstant(&arg2, MTLDataTypeUInt, 1),
  });
  auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
  [encoder setComputePipelineState:pso];
  MPSDispatchThreads(N, encoder, pso);
}

} // namespace

#define DEFINE_AFFINE_FUNC(T)                                                  \
  template <>                                                                  \
  void Affine<T, MPSContext>(                                                  \
      const int num_dims,                                                      \
      const int64_t* dims,                                                     \
      const int num_axes,                                                      \
      const int64_t* axes,                                                     \
      const T* x,                                                              \
      const T* scale,                                                          \
      const T* bias,                                                           \
      T* y,                                                                    \
      MPSContext* ctx) {                                                       \
    vec64_t new_dims, new_axes;                                                \
    math::utils::CollapseReduceAxes(                                           \
        num_dims, dims, num_axes, axes, new_dims, new_axes);                   \
    int bidx = 0;                                                              \
    auto* command_buffer = ctx->mps_stream()->command_buffer();                \
    auto* encoder = [command_buffer computeCommandEncoder];                    \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:bidx++];              \
    [encoder setBuffer:id<MTLBuffer>(scale) offset:0 atIndex:bidx++];          \
    if (bias) [encoder setBuffer:id<MTLBuffer>(bias) offset:0 atIndex:bidx++]; \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:bidx++];              \
    DispatchAffine<T>(                                                         \
        new_dims.size(),                                                       \
        new_dims.data(),                                                       \
        new_axes.size(),                                                       \
        new_axes.data(),                                                       \
        bias != nullptr,                                                       \
        encoder,                                                               \
        ctx);                                                                  \
    [encoder endEncoding];                                                     \
    [encoder release];                                                         \
  }

DEFINE_AFFINE_FUNC(float);
DEFINE_AFFINE_FUNC(float16);
DEFINE_AFFINE_FUNC(bfloat16);
DEFINE_AFFINE_FUNC(double);
#undef DEFINE_AFFINE_FUNC

} // namespace math

} // namespace dragon
