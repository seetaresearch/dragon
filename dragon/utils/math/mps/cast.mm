#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

template <typename InputT, typename OutputT>
kernel void Cast(
    device const InputT* x,
    device OutputT* y,
    const uint index [[thread_position_in_grid]]) {
  y[index] = OutputT(x[index]);
}

#define INSTANTIATE_CAST_KERNEL(InputT, OutputT) \
  template [[host_name("Cast_"#InputT"_"#OutputT)]] \
  kernel void Cast(device const InputT*, device OutputT*, uint);

#define INSTANTIATE_CAST_KERNEL_TO(OutputT) \
  INSTANTIATE_CAST_KERNEL(bool, OutputT); \
  INSTANTIATE_CAST_KERNEL(uint8_t, OutputT); \
  INSTANTIATE_CAST_KERNEL(int8_t, OutputT); \
  INSTANTIATE_CAST_KERNEL(int, OutputT); \
  INSTANTIATE_CAST_KERNEL(int64_t, OutputT); \
  INSTANTIATE_CAST_KERNEL(half, OutputT); \
  INSTANTIATE_CAST_KERNEL(float, OutputT);

INSTANTIATE_CAST_KERNEL_TO(bool);
INSTANTIATE_CAST_KERNEL_TO(uint8_t);
INSTANTIATE_CAST_KERNEL_TO(int8_t);
INSTANTIATE_CAST_KERNEL_TO(int);
INSTANTIATE_CAST_KERNEL_TO(int64_t);
INSTANTIATE_CAST_KERNEL_TO(half);
INSTANTIATE_CAST_KERNEL_TO(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_CAST_KERNEL_TO(double);
INSTANTIATE_CAST_KERNEL(double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_CAST_KERNEL
#undef INSTANTIATE_CAST_KERNEL_TO

)";

} // namespace

#define DEFINE_CAST_FUNC(InputT, OutputT)                          \
  template <>                                                      \
  DRAGON_API void Cast<InputT, OutputT, MPSContext>(               \
      const int N, const InputT* x, OutputT* y, MPSContext* ctx) { \
    auto kernel = MPSKernel::TypedString<InputT>("Cast");          \
    kernel = MPSKernel::TypedString<OutputT>(kernel);              \
    auto* command_buffer = ctx->mps_stream()->command_buffer();    \
    auto* encoder = [command_buffer computeCommandEncoder];        \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx);    \
    [encoder setComputePipelineState:pso];                         \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];       \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];       \
    MPSDispatchThreads(N, encoder, pso);                           \
    [encoder endEncoding];                                         \
    [encoder release];                                             \
  }

#define DEFINE_CAST_FUNC_TO(T)   \
  DEFINE_CAST_FUNC(T, bool);     \
  DEFINE_CAST_FUNC(T, uint8_t);  \
  DEFINE_CAST_FUNC(T, int8_t);   \
  DEFINE_CAST_FUNC(T, int);      \
  DEFINE_CAST_FUNC(T, int64_t);  \
  DEFINE_CAST_FUNC(T, float16);  \
  DEFINE_CAST_FUNC(T, bfloat16); \
  DEFINE_CAST_FUNC(T, float);    \
  DEFINE_CAST_FUNC(T, double);

DEFINE_CAST_FUNC_TO(bool);
DEFINE_CAST_FUNC_TO(uint8_t);
DEFINE_CAST_FUNC_TO(int8_t);
DEFINE_CAST_FUNC_TO(int);
DEFINE_CAST_FUNC_TO(int64_t);
DEFINE_CAST_FUNC_TO(float16);
DEFINE_CAST_FUNC_TO(bfloat16);
DEFINE_CAST_FUNC_TO(float);
DEFINE_CAST_FUNC_TO(double);
#undef DEFINE_CAST_FUNC
#undef DEFINE_CAST_FUNC_TO

} // namespace math

} // namespace dragon
