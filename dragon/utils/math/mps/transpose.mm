#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

#define kBlockRows 8

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

#define kBlockRows 8

template <typename T, int L, int N>
struct SimpleArray { vec<T, L> data[N]; };

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

#define DIVUP(a, b) (((a) - 1) / (b) + 1)

constant uint uint_arg1 [[function_constant(0)]]; // num_dims | H
constant uint uint_arg2 [[function_constant(1)]]; // W
constant uint4 uint4_arg1 [[function_constant(2)]];
constant uint4 uint4_arg2 [[function_constant(3)]];
constant uint4 uint4_arg3 [[function_constant(4)]];
constant uint4 uint4_arg4 [[function_constant(5)]];
constant SimpleArray<uint, 4, 2> uintarr_arg1 = {uint4_arg1, uint4_arg2};
constant SimpleArray<uint, 4, 2> uintarr_arg2 = {uint4_arg3, uint4_arg4};

template <typename T, int n>
kernel void AlignedTranspose(
    device const uint8_t* x,
    device uint8_t* y,
    const uint yi [[thread_position_in_grid]]) {
  typedef vec<T, n> ScalarT;
  device const ScalarT* x_aligned = (device const ScalarT*)x;
  device ScalarT* y_aligned = (device ScalarT*)y;
  uint xi = 0, tmp = yi, r;
  for (int d = uint_arg1 - 1; d >= 0; --d) {
    const int d1 = d / 4, d2 = d % 4;
    FIXED_DIVISOR_DIV_MOD(uintarr_arg2.data[d1][d2], tmp, &tmp, &r);
    xi += r * uintarr_arg1.data[d1][d2];
  }
  y_aligned[yi] = x_aligned[xi];
}

template <typename T>
kernel void BatchTranspose(
    device const T* X,
    device T* Y,
    threadgroup T* tile [[threadgroup(0)]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint block_id [[threadgroup_position_in_grid]],
    const uint warp_id [[simdgroup_index_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]]) {
  const uint dh = DIVUP(uint_arg1, warp_size);
  const uint dw = DIVUP(uint_arg2, warp_size);
  const uint k = block_id % (dh * dw);
  const uint r = k / dw, c = k % dw;
  const uint offset = block_id / (dh * dw) * uint_arg1 * uint_arg2;
  uint x = c * warp_size + lane_id, y = r * warp_size;
  if (x < uint_arg2) {
    for (uint i = warp_id; i < warp_size && y + i < uint_arg1; i += kBlockRows) {
      tile[i * warp_size + lane_id] = X[offset + (y + i) * uint_arg2 + x];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  x = r * warp_size + lane_id, y = c * warp_size;
  if (x < uint_arg1) {
    for (uint i = warp_id; i < warp_size && y + i < uint_arg2; i += kBlockRows) {
      Y[offset + (y + i) * uint_arg1 + x] = tile[lane_id * warp_size + i];
    }
  }
}

#define INSTANTIATE_KERNEL(T, n, L) \
  template [[host_name("AlignedTranspose_"#L "B")]] \
  kernel void AlignedTranspose<T, n>( \
      device const uint8_t*, device uint8_t*, uint);

INSTANTIATE_KERNEL(uchar, 1, 1);
INSTANTIATE_KERNEL(ushort, 1, 2);
INSTANTIATE_KERNEL(uint, 1, 4);
INSTANTIATE_KERNEL(uint, 2, 8);
INSTANTIATE_KERNEL(uint, 4, 16);
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_BATCH_KERNEL(T) \
  template [[host_name("BatchTranspose_"#T)]] \
  kernel void BatchTranspose( \
      device const T*, device T*, \
      threadgroup T*, uint, uint, uint, uint);

INSTANTIATE_BATCH_KERNEL(bool);
INSTANTIATE_BATCH_KERNEL(uint8_t);
INSTANTIATE_BATCH_KERNEL(int8_t);
INSTANTIATE_BATCH_KERNEL(int);
INSTANTIATE_BATCH_KERNEL(int64_t);
INSTANTIATE_BATCH_KERNEL(half);
INSTANTIATE_BATCH_KERNEL(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_BATCH_KERNEL(double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_BATCH_KERNEL

)";

template <typename T>
void DispatchTranspose(
    const vec64_t& dims,
    const vec64_t& axes,
    const T* x,
    T* y,
    MTLComputeCommandEncoder_t encoder,
    MPSContext* ctx) {
  const int num_dims = dims.size();
  MPS_TENSOR_DIMS_CHECK(num_dims);
  auto aligned_size = sizeof(T);
  if (axes.back() == num_dims - 1) {
    aligned_size = utils::GetAlignedSize<T, 16>(dims.back());
  }
  vec64_t X_dims(num_dims), X_strides(num_dims), Y_dims(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    X_dims[i] = dims[i];
  }
  X_dims[num_dims - 1] /= int64_t(aligned_size / sizeof(T));
  utils::ComputeTransposeStrides(
      num_dims, X_dims.data(), axes.data(), X_strides.data());
  const uint arg1 = num_dims;
  vector<uint32_t> arg2(MPS_TENSOR_MAX_DIMS, 0);
  vector<uint32_t> arg3(MPS_TENSOR_MAX_DIMS, 0);
  for (int i = 0; i < num_dims; ++i) {
    arg2[i] = X_strides[i];
    arg3[i] = X_dims[axes[i]];
  }
  auto kernel = "AlignedTranspose_" + str::to(aligned_size) + "B";
  auto args = vector<MPSConstant>(
      {MPSConstant(&arg1, MTLDataTypeUInt, 0),
       MPSConstant(arg2.data(), MTLDataTypeUInt4, {2, 3}),
       MPSConstant(arg3.data(), MTLDataTypeUInt4, {4, 5})});
  auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
  [encoder setComputePipelineState:pso];
  MPSDispatchThreads(math::utils::Prod(X_dims), encoder, pso);
}

template <typename T>
void DispatchBatchTranspose(
    const vec64_t& dims,
    const T* x,
    T* y,
    MTLComputeCommandEncoder_t encoder,
    MPSContext* ctx) {
  const uint arg1 = dims[1], arg2 = dims[2];
  auto kernel = MPSKernel::TypedString<T>("BatchTranspose");
  auto args = vector<MPSConstant>({
      MPSConstant(&arg1, MTLDataTypeUInt, 0),
      MPSConstant(&arg2, MTLDataTypeUInt, 1),
  });
  auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
  const int warp_size = pso.threadExecutionWidth; // 8, 16, 32, 64, ...
  const int storage_size = sizeof(T) * warp_size * (warp_size + 1);
  const int block_threads = warp_size * kBlockRows;
  const auto dh = math::utils::DivUp<int64_t>(dims[1], warp_size);
  const auto dw = math::utils::DivUp<int64_t>(dims[2], warp_size);
  const int num_blocks = dims[0] * dh * dw;
  [encoder setComputePipelineState:pso];
  [encoder setThreadgroupMemoryLength:storage_size atIndex:0];
  MPSDispatchThreads(num_blocks, block_threads, encoder, pso);
}

} // namespace

#define DEFINE_TRANSPOSE_FUNC(T)                                            \
  template <>                                                               \
  DRAGON_API void Transpose<T, MPSContext>(                                 \
      const int num_dims,                                                   \
      const int64_t* dims,                                                  \
      const int64_t* axes,                                                  \
      const T* x,                                                           \
      T* y,                                                                 \
      MPSContext* ctx) {                                                    \
    vec64_t new_dims, new_axes;                                             \
    utils::CollapseTransposeAxes(num_dims, dims, axes, new_dims, new_axes); \
    auto* command_buffer = ctx->mps_stream()->command_buffer();             \
    auto* encoder = [command_buffer computeCommandEncoder];                 \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];                \
    const int num_axes = new_dims.size();                                   \
    if (num_axes == 3 && new_axes == vec64_t({0, 2, 1})) {                  \
      DispatchBatchTranspose(new_dims, x, y, encoder, ctx);                 \
    } else {                                                                \
      DispatchTranspose(new_dims, new_axes, x, y, encoder, ctx);            \
    }                                                                       \
    [encoder endEncoding];                                                  \
    [encoder release];                                                      \
  }

DEFINE_TRANSPOSE_FUNC(bool);
DEFINE_TRANSPOSE_FUNC(uint8_t);
DEFINE_TRANSPOSE_FUNC(int8_t);
DEFINE_TRANSPOSE_FUNC(int);
DEFINE_TRANSPOSE_FUNC(int64_t);
DEFINE_TRANSPOSE_FUNC(float16);
DEFINE_TRANSPOSE_FUNC(bfloat16);
DEFINE_TRANSPOSE_FUNC(float);
DEFINE_TRANSPOSE_FUNC(double);
#undef DEFINE_TRANSPOSE_FUNC

} // namespace math

} // namespace dragon
