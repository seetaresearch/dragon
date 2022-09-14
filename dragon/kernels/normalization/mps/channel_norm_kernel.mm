#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

template <typename T, int L, int N>
struct SimpleArray { vec<T, L> data[N]; };

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

constant uint uint_arg1 [[function_constant(0)]]; // num_dims
constant int int_arg1 [[function_constant(1)]];   // axis
constant uint4 uint4_arg1 [[function_constant(2)]];
constant uint4 uint4_arg2 [[function_constant(3)]];
constant uint4 uint4_arg3 [[function_constant(4)]];
constant uint4 uint4_arg4 [[function_constant(5)]];
constant SimpleArray<uint, 4, 2> uintarr_arg1 = {uint4_arg1, uint4_arg2}; // X_strides
constant SimpleArray<uint, 4, 2> uintarr_arg2 = {uint4_arg3, uint4_arg4}; // Y_dims

template <typename InputT, typename OutputT>
kernel void ChannelNorm(
    device const InputT* x,
    device const float* mean,
    device const float* std,
    device OutputT* y,
    const uint yi [[thread_position_in_grid]]) {
  uint xi = 0, wi, tmp = yi, r;
  for (int d = uint_arg1 - 1; d >= 0; --d) {
    const int d1 = d / 4, d2 = d % 4;
    FIXED_DIVISOR_DIV_MOD(uintarr_arg2.data[d1][d2], tmp, &tmp, &r);
    xi += r * uintarr_arg1.data[d1][d2];
    if (d == int_arg1) wi = r;
  }
  y[yi] = OutputT((float(x[xi]) - mean[wi]) / std[wi]);
}

#define INSTANTIATE_KERNEL(InputT, OutputT) \
  template [[host_name("ChannelNorm_"#InputT"_"#OutputT)]] \
  kernel void ChannelNorm(device const InputT*, device const float*, \
                          device const float*, device OutputT*, uint);

INSTANTIATE_KERNEL(uint8_t, half);
INSTANTIATE_KERNEL(uint8_t, float);
INSTANTIATE_KERNEL(int8_t, half);
INSTANTIATE_KERNEL(int8_t, float);
INSTANTIATE_KERNEL(int, half);
INSTANTIATE_KERNEL(int, float);
INSTANTIATE_KERNEL(int64_t, half);
INSTANTIATE_KERNEL(int64_t, float);
INSTANTIATE_KERNEL(half, half);
INSTANTIATE_KERNEL(half, float);
INSTANTIATE_KERNEL(float, half);
INSTANTIATE_KERNEL(float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(uint8_t, double);
INSTANTIATE_KERNEL(int8_t, double);
INSTANTIATE_KERNEL(int, double);
INSTANTIATE_KERNEL(int64_t, double);
INSTANTIATE_KERNEL(half, double);
INSTANTIATE_KERNEL(float, double);
INSTANTIATE_KERNEL(double, half);
INSTANTIATE_KERNEL(double, float);
INSTANTIATE_KERNEL(double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(InputT, OutputT)                            \
  template <>                                                              \
  void ChannelNorm<InputT, OutputT, MPSContext>(                           \
      const int axis,                                                      \
      const int num_dims,                                                  \
      const int64_t* x_strides,                                            \
      const int64_t* y_dims,                                               \
      const InputT* x,                                                     \
      const float* mean,                                                   \
      const float* std,                                                    \
      OutputT* y,                                                          \
      MPSContext* ctx) {                                                   \
    MPS_TENSOR_DIMS_CHECK(num_dims);                                       \
    const uint arg1 = num_dims;                                            \
    vector<uint32_t> arg3(MPS_TENSOR_MAX_DIMS, 0);                         \
    vector<uint32_t> arg4(MPS_TENSOR_MAX_DIMS, 0);                         \
    for (int i = 0; i < num_dims; ++i) {                                   \
      arg3[i] = x_strides[i];                                              \
      arg4[i] = y_dims[i];                                                 \
    }                                                                      \
    auto kernel = MPSKernel::TypedString<InputT>("ChannelNorm");           \
    kernel = MPSKernel::TypedString<OutputT>(kernel);                      \
    auto args = vector<MPSConstant>(                                       \
        {MPSConstant(&arg1, MTLDataTypeUInt, 0),                           \
         MPSConstant(&axis, MTLDataTypeInt, 1),                            \
         MPSConstant(arg3.data(), MTLDataTypeUInt4, {2, 3}),               \
         MPSConstant(arg4.data(), MTLDataTypeUInt4, {4, 5})});             \
    auto* command_buffer = ctx->mps_stream()->command_buffer();            \
    auto* encoder = [command_buffer computeCommandEncoder];                \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);      \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];               \
    [encoder setBuffer:id<MTLBuffer>(mean) offset:0 atIndex:1];            \
    [encoder setBuffer:id<MTLBuffer>(std) offset:0 atIndex:2];             \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:3];               \
    [encoder setComputePipelineState:pso];                                 \
    MPSDispatchThreads(math::utils::Prod(num_dims, y_dims), encoder, pso); \
    [encoder endEncoding];                                                 \
    [encoder release];                                                     \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t, float16);
DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(uint8_t, double);
DEFINE_KERNEL_LAUNCHER(int8_t, float16);
DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(int8_t, double);
DEFINE_KERNEL_LAUNCHER(int, float16);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int, double);
DEFINE_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float16, float16);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float16, double);
DEFINE_KERNEL_LAUNCHER(float, float16);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(float, double);
DEFINE_KERNEL_LAUNCHER(double, float16);
DEFINE_KERNEL_LAUNCHER(double, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
