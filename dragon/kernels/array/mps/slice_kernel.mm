#include "dragon/kernels/array/op_kernels.h"
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
constant uint4 uint4_arg1 [[function_constant(1)]];
constant uint4 uint4_arg2 [[function_constant(2)]];
constant uint4 uint4_arg3 [[function_constant(3)]];
constant uint4 uint4_arg4 [[function_constant(4)]];
constant uint4 uint4_arg5 [[function_constant(5)]];
constant uint4 uint4_arg6 [[function_constant(6)]];
constant SimpleArray<uint, 4, 2> uintarr_arg1 = {uint4_arg1, uint4_arg2};
constant SimpleArray<uint, 4, 2> uintarr_arg2 = {uint4_arg3, uint4_arg4};
constant SimpleArray<uint, 4, 2> uintarr_arg3 = {uint4_arg5, uint4_arg6};

template <typename T>
kernel void Slice(
    device const T* x,
    device T* y,
    const uint yi [[thread_position_in_grid]]) {
  uint xi = 0, tmp = yi, r;
  for (int d = uint_arg1 - 1; d >= 0; --d) {
    const int d1 = d / 4, d2 = d % 4;
    FIXED_DIVISOR_DIV_MOD(uintarr_arg2.data[d1][d2], tmp, &tmp, &r);
    xi += (r + uintarr_arg3.data[d1][d2]) * uintarr_arg1.data[d1][d2];
  }
  y[yi] = x[xi];
}

template <typename T>
kernel void SliceGrad(
    device const T* dy,
    device T* dx,
    const uint yi [[thread_position_in_grid]]) {
  uint xi = 0, tmp = yi, r;
  for (int d = uint_arg1 - 1; d >= 0; --d) {
    const int d1 = d / 4, d2 = d % 4;
    FIXED_DIVISOR_DIV_MOD(uintarr_arg2.data[d1][d2], tmp, &tmp, &r);
    xi += (r + uintarr_arg3.data[d1][d2]) * uintarr_arg1.data[d1][d2];
  }
  dx[xi] = dy[yi];
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device T*, uint);

INSTANTIATE_KERNEL(Slice, bool);
INSTANTIATE_KERNEL(Slice, uint8_t);
INSTANTIATE_KERNEL(Slice, int8_t);
INSTANTIATE_KERNEL(Slice, int);
INSTANTIATE_KERNEL(Slice, int64_t);
INSTANTIATE_KERNEL(Slice, half);
INSTANTIATE_KERNEL(Slice, float);
INSTANTIATE_KERNEL(SliceGrad, bool);
INSTANTIATE_KERNEL(SliceGrad, uint8_t);
INSTANTIATE_KERNEL(SliceGrad, int8_t);
INSTANTIATE_KERNEL(SliceGrad, int);
INSTANTIATE_KERNEL(SliceGrad, int64_t);
INSTANTIATE_KERNEL(SliceGrad, half);
INSTANTIATE_KERNEL(SliceGrad, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(Slice, double);
INSTANTIATE_KERNEL(SliceGrad, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

template <typename T>
void DispatchKernel(
    const string& name,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* starts,
    const T* x,
    T* y,
    MTLComputeCommandEncoder_t encoder,
    MPSContext* ctx) {
  MPS_TENSOR_DIMS_CHECK(num_dims);
  const uint arg1 = num_dims;
  vector<uint32_t> arg2(MPS_TENSOR_MAX_DIMS, 0);
  vector<uint32_t> arg3(MPS_TENSOR_MAX_DIMS, 0);
  vector<uint32_t> arg4(MPS_TENSOR_MAX_DIMS, 0);
  for (int i = 0; i < num_dims; ++i) {
    arg2[i] = x_strides[i], arg3[i] = y_dims[i], arg4[i] = starts[i];
  }
  auto kernel = MPSKernel::TypedString<T>(name);
  auto args = vector<MPSConstant>(
      {MPSConstant(&arg1, MTLDataTypeUInt, 0),
       MPSConstant(arg2.data(), MTLDataTypeUInt4, {1, 2}),
       MPSConstant(arg3.data(), MTLDataTypeUInt4, {3, 4}),
       MPSConstant(arg4.data(), MTLDataTypeUInt4, {5, 6})});
  auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
  [encoder setComputePipelineState:pso];
  MPSDispatchThreads(math::utils::Prod(num_dims, y_dims), encoder, pso);
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                  \
  template <>                                                            \
  void name<T, MPSContext>(                                              \
      const int num_dims,                                                \
      const int64_t* x_strides,                                          \
      const int64_t* y_dims,                                             \
      const int64_t* starts,                                             \
      const T* x,                                                        \
      T* y,                                                              \
      MPSContext* ctx) {                                                 \
    auto* command_buffer = ctx->mps_stream()->command_buffer();          \
    auto* encoder = [command_buffer computeCommandEncoder];              \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];             \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];             \
    DispatchKernel(                                                      \
        #name, num_dims, x_strides, y_dims, starts, x, y, encoder, ctx); \
    [encoder endEncoding];                                               \
    [encoder release];                                                   \
  }

DEFINE_KERNEL_LAUNCHER(Slice, bool);
DEFINE_KERNEL_LAUNCHER(Slice, int8_t);
DEFINE_KERNEL_LAUNCHER(Slice, uint8_t);
DEFINE_KERNEL_LAUNCHER(Slice, int);
DEFINE_KERNEL_LAUNCHER(Slice, int64_t);
DEFINE_KERNEL_LAUNCHER(Slice, float16);
DEFINE_KERNEL_LAUNCHER(Slice, bfloat16);
DEFINE_KERNEL_LAUNCHER(Slice, float);
DEFINE_KERNEL_LAUNCHER(Slice, double);
DEFINE_KERNEL_LAUNCHER(SliceGrad, float16);
DEFINE_KERNEL_LAUNCHER(SliceGrad, bfloat16);
DEFINE_KERNEL_LAUNCHER(SliceGrad, float);
DEFINE_KERNEL_LAUNCHER(SliceGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
