#include "dragon/kernels/math/op_kernels.h"
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

constant uint uint_arg1 [[function_constant(0)]];
constant float float_arg1 [[function_constant(1)]];
constant uint4 uint4_arg1 [[function_constant(2)]];
constant uint4 uint4_arg2 [[function_constant(3)]];
constant uint4 uint4_arg3 [[function_constant(4)]];
constant uint4 uint4_arg4 [[function_constant(5)]];
constant uint4 uint4_arg5 [[function_constant(6)]];
constant uint4 uint4_arg6 [[function_constant(7)]];
constant SimpleArray<uint, 4, 2> uintarr_arg1 = {uint4_arg1, uint4_arg2};
constant SimpleArray<uint, 4, 2> uintarr_arg2 = {uint4_arg3, uint4_arg4};
constant SimpleArray<uint, 4, 2> uintarr_arg3 = {uint4_arg5, uint4_arg6};

template <typename T>
kernel void ReduceSumGrad(
    device const T* dy,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  uint yi = 0, tmp = index, r;
  for (int d = uint_arg1 - 1; d >= 0; --d) {
    const int d1 = d / 4, d2 = d % 4;
    FIXED_DIVISOR_DIV_MOD(uintarr_arg1.data[d1][d2], tmp, &tmp, &r);
    yi += (r % uintarr_arg2.data[d1][d2]) * uintarr_arg3.data[d1][d2];
  }
  dx[index] = dy[yi] * T(float_arg1);
}

#define INSTANTIATE_GRAD_KERNEL(T) \
  template [[host_name("ReduceSumGrad_"#T)]] \
  kernel void ReduceSumGrad(device const T*, device T*, uint);

INSTANTIATE_GRAD_KERNEL(half);
INSTANTIATE_GRAD_KERNEL(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                     \
  template <>                                                              \
  void ReduceSumGrad<T, MPSContext>(                                       \
      const int num_dims,                                                  \
      const int64_t* x_dims,                                               \
      const int64_t* y_dims,                                               \
      const int64_t* y_strides,                                            \
      const float scale,                                                   \
      const T* dy,                                                         \
      T* dx,                                                               \
      MPSContext* ctx) {                                                   \
    MPS_TENSOR_DIMS_CHECK(num_dims);                                       \
    auto kernel = MPSKernel::TypedString<T>("ReduceSumGrad");              \
    const uint arg1 = num_dims;                                            \
    vector<uint32_t> arg3(MPS_TENSOR_MAX_DIMS, 0);                         \
    vector<uint32_t> arg4(MPS_TENSOR_MAX_DIMS, 0);                         \
    vector<uint32_t> arg5(MPS_TENSOR_MAX_DIMS, 0);                         \
    for (int i = 0; i < num_dims; ++i) {                                   \
      arg3[i] = x_dims[i], arg4[i] = y_dims[i], arg5[i] = y_strides[i];    \
    }                                                                      \
    const auto args = vector<MPSConstant>({                                \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                            \
        MPSConstant(&scale, MTLDataTypeFloat, 1),                          \
        MPSConstant(arg3.data(), MTLDataTypeUInt4, {2, 3}),                \
        MPSConstant(arg4.data(), MTLDataTypeUInt4, {4, 5}),                \
        MPSConstant(arg5.data(), MTLDataTypeUInt4, {6, 7}),                \
    });                                                                    \
    auto* command_buffer = ctx->mps_stream()->command_buffer();            \
    auto* encoder = [command_buffer computeCommandEncoder];                \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);      \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:0];              \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:1];              \
    [encoder setComputePipelineState:pso];                                 \
    MPSDispatchThreads(math::utils::Prod(num_dims, x_dims), encoder, pso); \
    [encoder endEncoding];                                                 \
    [encoder release];                                                     \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
