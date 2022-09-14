#include "dragon/kernels/normalization/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // C
constant uint uint_arg2 [[function_constant(1)]]; // S
constant float float_arg1 [[function_constant(2)]]; // normalizer
constant float float_arg2 [[function_constant(3)]]; // epsilon

template <typename T>
T Square(const T x) {
  return x * x;
}

template <typename T>
T WarpReduceSum(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val += simd_shuffle_down(val, offset);
  }
  return val;
}

template <typename T, typename AccT>
kernel void L1Norm(
    device const T* x,
    device T* y,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_norm;
  const uint offset = i / uint_arg2 * uint_arg1 * uint_arg2 + i % uint_arg2;
  // BlockReduceSum
  AccT val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val += abs(AccT(x[offset + j * uint_arg2]));
  }
  val = WarpReduceSum(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < (block_size / warp_size) ? storage[jstart] : AccT(0);
  val = WarpReduceSum(val, warp_size);
  if (jstart == 0) {
    block_norm = max(val / AccT(float_arg1), AccT(float_arg2));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssgin
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint index = offset + j * uint_arg2;
    y[index] = T(AccT(x[index]) / block_norm);
  }
}

template <typename T, typename AccT>
kernel void L2Norm(
    device const T* x,
    device T* y,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_norm;
  const uint offset = i / uint_arg2 * uint_arg1 * uint_arg2 + i % uint_arg2;
  // BlockReduceSum
  AccT val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val += Square(AccT(x[offset + j * uint_arg2]));
  }
  val = WarpReduceSum(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < (block_size / warp_size) ? storage[jstart] : AccT(0);
  val = WarpReduceSum(val, warp_size);
  if (jstart == 0) {
    block_norm = max(sqrt(val / AccT(float_arg1)), AccT(float_arg2));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssgin
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint index = offset + j * uint_arg2;
    y[index] = T(AccT(x[index]) / block_norm);
  }
}

template <typename T, typename AccT>
kernel void L1NormGrad(
    device const T* dy,
    device const T* x,
    device T* dx,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_norm1, block_norm2, block_sum;
  const uint block_warps = block_size / warp_size;
  const uint offset = i / uint_arg2 * uint_arg1 * uint_arg2 + i % uint_arg2;
  threadgroup AccT* storage1 = storage;
  threadgroup AccT* storage2 = storage + block_warps;
  // BlockReduceSum
  AccT val1 = AccT(0), val2 = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint index = offset + j * uint_arg2;
    val1 += abs(AccT(x[index]));
    val2 += AccT(dy[index]) * AccT(x[index]);
  }
  val1 = WarpReduceSum(val1, warp_size);
  val2 = WarpReduceSum(val2, warp_size);
  if (lane_id == 0) {
    storage1[warp_id] = val1;
    storage2[warp_id] = val2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val1 = jstart < block_warps ? storage1[jstart] : AccT(0);
  val2 = jstart < block_warps ? storage2[jstart] : AccT(0);
  val1 = WarpReduceSum(val1, warp_size);
  val2 = WarpReduceSum(val2, warp_size);
  if (jstart == 0) {
    block_norm1 = max(val1 / AccT(float_arg1), AccT(float_arg2));
    block_norm2 = block_norm1 * block_norm1;
    block_sum = val2 / AccT(float_arg1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssgin
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint index = offset + j * uint_arg2;
    dx[index] = T((AccT(dy[index]) / block_norm1) -
                  (sign(AccT(x[index])) / block_norm2 * block_sum));
  }
}

template <typename T, typename AccT>
kernel void L2NormGrad(
    device const T* dy,
    device const T* x,
    device T* dx,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_norm1, block_norm2, block_sum;
  const uint block_warps = block_size / warp_size;
  const uint offset = i / uint_arg2 * uint_arg1 * uint_arg2 + i % uint_arg2;
  threadgroup AccT* storage1 = storage;
  threadgroup AccT* storage2 = storage + block_warps;
  // BlockReduceSum
  AccT val1 = AccT(0), val2 = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint index = offset + j * uint_arg2;
    val1 += Square(AccT(x[index]));
    val2 += AccT(dy[index]) * AccT(x[index]);
  }
  val1 = WarpReduceSum(val1, warp_size);
  val2 = WarpReduceSum(val2, warp_size);
  if (lane_id == 0) {
    storage1[warp_id] = val1;
    storage2[warp_id] = val2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val1 = jstart < block_warps ? storage1[jstart] : AccT(0);
  val2 = jstart < block_warps ? storage2[jstart] : AccT(0);
  val1 = WarpReduceSum(val1, warp_size);
  val2 = WarpReduceSum(val2, warp_size);
  if (jstart == 0) {
    block_norm1 = max(sqrt(val1 / AccT(float_arg1)), AccT(float_arg2));
    block_norm2 = block_norm1 * block_norm1 * block_norm1;
    block_sum = val2 / AccT(float_arg1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssgin
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint index = offset + j * uint_arg2;
    dx[index] = T((AccT(dy[index]) / block_norm1) -
                  (AccT(x[index]) / block_norm2 * block_sum));
  }
}

#define INSTANTIATE_KERNEL(name, T, AccT) \
  template [[host_name(#name"_"#T)]] \
  kernel void name<T, AccT>( \
      device const T*, device T*, \
      threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_KERNEL(L1Norm, half, float);
INSTANTIATE_KERNEL(L1Norm, float, float);
INSTANTIATE_KERNEL(L2Norm, half, float);
INSTANTIATE_KERNEL(L2Norm, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(L1Norm, double, double);
INSTANTIATE_KERNEL(L2Norm, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, T, AccT) \
  template [[host_name(#name"_"#T)]] \
  kernel void name<T, AccT>( \
      device const T*, device const T*, device T*, \
      threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_GRAD_KERNEL(L1NormGrad, half, float);
INSTANTIATE_GRAD_KERNEL(L1NormGrad, float, float);
INSTANTIATE_GRAD_KERNEL(L2NormGrad, half, float);
INSTANTIATE_GRAD_KERNEL(L2NormGrad, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(L1NormGrad, double, double);
INSTANTIATE_GRAD_KERNEL(L2NormGrad, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

)";

template <typename T, typename AccT>
void DispatchKernel(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const string& name,
    MTLComputeCommandEncoder_t encoder,
    MPSContext* ctx) {
  const uint arg1 = C, arg2 = S;
  auto args = vector<MPSConstant>({
      MPSConstant(&arg1, MTLDataTypeUInt, 0),
      MPSConstant(&arg2, MTLDataTypeUInt, 1),
      MPSConstant(&normalizer, MTLDataTypeFloat, 2),
      MPSConstant(&epsilon, MTLDataTypeFloat, 3),
  });
  auto kernel = MPSKernel::TypedString<T>(name);
  auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
  const int block_threads = MPSGetBlockReduceThreads(C, pso);
  const int block_warps = block_threads / pso.threadExecutionWidth;
  auto storage_size = block_warps * sizeof(AccT);
  if (name.find("Grad") != string::npos) storage_size *= 2;
  [encoder setComputePipelineState:pso];
  [encoder setThreadgroupMemoryLength:storage_size atIndex:0];
  MPSDispatchThreads((N * S), block_threads, encoder, pso);
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, AccT)                   \
  template <>                                                   \
  void name<T, MPSContext>(                                     \
      const int N,                                              \
      const int S,                                              \
      const int C,                                              \
      const float normalizer,                                   \
      const float epsilon,                                      \
      const T* x,                                               \
      T* y,                                                     \
      MPSContext* ctx) {                                        \
    auto* command_buffer = ctx->mps_stream()->command_buffer(); \
    auto* encoder = [command_buffer computeCommandEncoder];     \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];    \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];    \
    DispatchKernel<T, AccT>(                                    \
        N, S, C, normalizer, epsilon, #name, encoder, ctx);     \
    [encoder endEncoding];                                      \
    [encoder release];                                          \
  }

DEFINE_KERNEL_LAUNCHER(L1Norm, float16, float);
DEFINE_KERNEL_LAUNCHER(L1Norm, float, float);
DEFINE_KERNEL_LAUNCHER(L1Norm, double, double);
DEFINE_KERNEL_LAUNCHER(L2Norm, float16, float);
DEFINE_KERNEL_LAUNCHER(L2Norm, float, float);
DEFINE_KERNEL_LAUNCHER(L2Norm, double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T, AccT)              \
  template <>                                                   \
  void name<T, MPSContext>(                                     \
      const int N,                                              \
      const int S,                                              \
      const int C,                                              \
      const float normalizer,                                   \
      const float epsilon,                                      \
      const T* dy,                                              \
      const T* x,                                               \
      T* dx,                                                    \
      MPSContext* ctx) {                                        \
    auto* command_buffer = ctx->mps_stream()->command_buffer(); \
    auto* encoder = [command_buffer computeCommandEncoder];     \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:0];   \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:1];    \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:2];   \
    DispatchKernel<T, AccT>(                                    \
        N, S, C, normalizer, epsilon, #name, encoder, ctx);     \
    [encoder endEncoding];                                      \
    [encoder release];                                          \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L1NormGrad, double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(L2NormGrad, double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
