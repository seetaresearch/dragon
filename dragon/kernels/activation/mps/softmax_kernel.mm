#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // C
constant uint uint_arg2 [[function_constant(1)]]; // S
constant uint uint_arg3 [[function_constant(2)]]; // N

template <typename T>
T WarpReduceMax(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val = max(val, simd_shuffle_down(val, offset));
  }
  return val;
}

template <typename T>
T WarpAllReduceMax(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val = max(val, simd_shuffle_xor(val, offset));
  }
  return val;
}

template <typename T>
T WarpReduceSum(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val += simd_shuffle_down(val, offset);
  }
  return val;
}

template <typename T>
T WarpAllReduceSum(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val += simd_shuffle_xor(val, offset);
  }
  return val;
}

template <typename T, typename AccT>
kernel void WarpSoftmax(
    device const T* x,
    device T* y,
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint block_id [[threadgroup_position_in_grid]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = block_id * (block_size / warp_size) + warp_id;
  if (i >= (uint_arg2 * uint_arg3)) return;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_y = y + offset;
  device const T* offset_x = x + offset;
  AccT val = AccT(-FLT_MAX);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    val = max(val, AccT(offset_x[j * uint_arg2]));
  }
  const AccT warp_max = WarpAllReduceMax(val, warp_size);
  val = AccT(0);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    val += exp(offset_x[j * uint_arg2] - warp_max);
  }
  const AccT warp_sum = WarpAllReduceSum(val, warp_size);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    const uint k = j * uint_arg2;
    val = exp(offset_x[k] - warp_max);
    offset_y[k] = T(val / warp_sum);
  }
}

template <typename T, typename AccT>
kernel void BlockSoftmax(
    device const T* x,
    device T* y,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_max, block_sum;
  const uint block_warps = block_size / warp_size;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_y = y + offset;
  device const T* offset_x = x + offset;
  // BlockReduceMax
  AccT val = AccT(-FLT_MAX);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val = max(val, AccT(offset_x[j * uint_arg2]));
  }
  val = WarpReduceMax(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < block_warps ? storage[jstart] : AccT(-FLT_MAX);
  val = WarpReduceMax(val, warp_size);
  if (jstart == 0) block_max = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockReduceSum
  val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val += exp(offset_x[j * uint_arg2] - block_max);
  }
  val = WarpReduceSum(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < block_warps ? storage[jstart] : AccT(0);
  val = WarpReduceSum(val, warp_size);
  if (jstart == 0) block_sum = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssign
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint k = j * uint_arg2;
    val = exp(offset_x[k] - block_max);
    offset_y[k] = T(val / block_sum);
  }
}

template <typename T, typename AccT>
kernel void WarpLogSoftmax(
    device const T* x,
    device T* y,
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint block_id [[threadgroup_position_in_grid]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = block_id * (block_size / warp_size) + warp_id;
  if (i >= (uint_arg2 * uint_arg3)) return;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_y = y + offset;
  device const T* offset_x = x + offset;
  AccT val = AccT(-FLT_MAX);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    val = max(val, AccT(offset_x[j * uint_arg2]));
  }
  const AccT warp_max = WarpAllReduceMax(val, warp_size);
  val = AccT(0);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    val += exp(offset_x[j * uint_arg2] - warp_max);
  }
  const AccT warp_sum = WarpAllReduceSum(val, warp_size);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    const uint k = j * uint_arg2;
    val = offset_x[k] - warp_max;
    offset_y[k] = T(val - log(warp_sum));
  }
}

template <typename T, typename AccT>
kernel void BlockLogSoftmax(
    device const T* x,
    device T* y,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_max, block_sum;
  const uint block_warps = block_size / warp_size;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_y = y + offset;
  device const T* offset_x = x + offset;
  // BlockReduceMax
  AccT val = AccT(-FLT_MAX);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val = max(val, AccT(offset_x[j * uint_arg2]));
  }
  val = WarpReduceMax(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < block_warps ? storage[jstart] : AccT(-FLT_MAX);
  val = WarpReduceMax(val, warp_size);
  if (jstart == 0) block_max = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockReduceSum
  val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val += exp(offset_x[j * uint_arg2] - block_max);
  }
  val = WarpReduceSum(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < block_warps ? storage[jstart] : AccT(0);
  val = WarpReduceSum(val, warp_size);
  if (jstart == 0) block_sum = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssign
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint k = j * uint_arg2;
    val = offset_x[k] - block_max;
    offset_y[k] = T(val - log(block_sum));
  }
}

template <typename T, typename AccT>
kernel void WarpSoftmaxGrad(
    device const T* dy,
    device const T* y,
    device T* dx,
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint block_id [[threadgroup_position_in_grid]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = block_id * (block_size / warp_size) + warp_id;
  if (i >= (uint_arg2 * uint_arg3)) return;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_dx = dx + offset;
  device const T* offset_y = y + offset;
  device const T* offset_dy = dy + offset;
  AccT val = AccT(0);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    const uint k = j * uint_arg2;
    val += AccT(offset_dy[k] * offset_y[k]);
  }
  const AccT warp_sum = WarpAllReduceSum(val, warp_size);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    const uint k = j * uint_arg2;
    val = AccT(offset_dy[k]) - warp_sum;
    offset_dx[k] = T(val * AccT(offset_y[k]));
  }
}

template <typename T, typename AccT>
kernel void BlockSoftmaxGrad(
    device const T* dy,
    device const T* y,
    device T* dx,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_sum;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_dx = dx + offset;
  device const T* offset_y = y + offset;
  device const T* offset_dy = dy + offset;
  // BlockReduceSum
  AccT val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint k = j * uint_arg2;
    val += AccT(offset_dy[k] * offset_y[k]);
  }
  val = WarpReduceSum(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < (block_size / warp_size) ? storage[jstart] : AccT(0);
  val = WarpReduceSum(val, warp_size);
  if (jstart == 0) block_sum = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssign
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint k = j * uint_arg2;
    val = AccT(offset_dy[k]) - block_sum;
    offset_dx[k] = T(val * AccT(offset_y[k]));
  }
}

template <typename T, typename AccT>
kernel void WarpLogSoftmaxGrad(
    device const T* dy,
    device const T* y,
    device T* dx,
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint block_id [[threadgroup_position_in_grid]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = block_id * (block_size / warp_size) + warp_id;
  if (i >= (uint_arg2 * uint_arg3)) return;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_dx = dx + offset;
  device const T* offset_y = y + offset;
  device const T* offset_dy = dy + offset;
  AccT val = AccT(0);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    val += AccT(offset_dy[j * uint_arg2]);
  }
  const AccT warp_sum = WarpAllReduceSum(val, warp_size);
  for (uint j = lane_id; j < uint_arg1; j += warp_size) {
    const uint k = j * uint_arg2;
    val = exp(AccT(offset_y[k])) * warp_sum;
    offset_dx[k] = T(AccT(offset_dy[k]) - val);
  }
}

template <typename T, typename AccT>
kernel void BlockLogSoftmaxGrad(
    device const T* dy,
    device const T* y,
    device T* dx,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup AccT block_sum;
  const uint offset = (i / uint_arg2) * uint_arg1 * uint_arg2 + i % uint_arg2;
  device T* offset_dx = dx + offset;
  device const T* offset_y = y + offset;
  device const T* offset_dy = dy + offset;
  // BlockReduceSum
  AccT val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val += AccT(offset_dy[j * uint_arg2]);
  }
  val = WarpReduceSum(val, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < (block_size / warp_size) ? storage[jstart] : AccT(0);
  val = WarpReduceSum(val, warp_size);
  if (jstart == 0) block_sum = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // BlockAssign
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint k = j * uint_arg2;
    val = exp(AccT(offset_y[k])) * block_sum;
    offset_dx[k] = T(AccT(offset_dy[k]) - val);
  }
}

#define INSTANTIATE_KERNEL(name, T, AccT) \
  template [[host_name(#name"_"#T)]] \
  kernel void name<T, AccT>( \
      device const T*, device T*, uint, uint, uint, uint, uint);

INSTANTIATE_KERNEL(WarpSoftmax, half, float);
INSTANTIATE_KERNEL(WarpSoftmax, float, float);
INSTANTIATE_KERNEL(WarpLogSoftmax, half, float);
INSTANTIATE_KERNEL(WarpLogSoftmax, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(WarpSoftmax, double, double);
INSTANTIATE_KERNEL(WarpLogSoftmax, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_KERNEL(name, T, AccT) \
  template [[host_name(#name"_"#T)]] \
  kernel void name<T, AccT>( \
      device const T*, device T*, \
      threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_KERNEL(BlockSoftmax, half, float);
INSTANTIATE_KERNEL(BlockSoftmax, float, float);
INSTANTIATE_KERNEL(BlockLogSoftmax, half, float);
INSTANTIATE_KERNEL(BlockLogSoftmax, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(BlockSoftmax, double, double);
INSTANTIATE_KERNEL(BlockLogSoftmax, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, T, AccT) \
  template [[host_name(#name"_"#T)]] \
  kernel void name<T, AccT>( \
      device const T*, device const T*, device T*, \
      uint, uint, uint, uint, uint);

INSTANTIATE_GRAD_KERNEL(WarpSoftmaxGrad, half, float);
INSTANTIATE_GRAD_KERNEL(WarpSoftmaxGrad, float, float);
INSTANTIATE_GRAD_KERNEL(WarpLogSoftmaxGrad, half, float);
INSTANTIATE_GRAD_KERNEL(WarpLogSoftmaxGrad, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(WarpSoftmaxGrad, double, double);
INSTANTIATE_GRAD_KERNEL(WarpLogSoftmaxGrad, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, T, AccT) \
  template [[host_name(#name"_"#T)]] \
  kernel void name<T, AccT>( \
      device const T*, device const T*, device T*, \
      threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_GRAD_KERNEL(BlockSoftmaxGrad, half, float);
INSTANTIATE_GRAD_KERNEL(BlockSoftmaxGrad, float, float);
INSTANTIATE_GRAD_KERNEL(BlockLogSoftmaxGrad, half, float);
INSTANTIATE_GRAD_KERNEL(BlockLogSoftmaxGrad, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(BlockSoftmaxGrad, double, double);
INSTANTIATE_GRAD_KERNEL(BlockLogSoftmaxGrad, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

)";

template <typename T>
void DispatchKernel(
    const int N,
    const int S,
    const int C,
    const string& name,
    MTLComputeCommandEncoder_t encoder,
    MPSContext* ctx) {
  const int NxS = N * S;
  const uint arg1 = C, arg2 = S, arg3 = N;
  auto args = vector<MPSConstant>({
      MPSConstant(&arg1, MTLDataTypeUInt, 0),
      MPSConstant(&arg2, MTLDataTypeUInt, 1),
      MPSConstant(&arg3, MTLDataTypeUInt, 2),
  });
  MTLComputePipelineState_t pso = nil;
  if (C <= 1024) {
    auto kernel = MPSKernel::TypedString<T>("Warp" + name);
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int warp_size = pso.threadExecutionWidth; // 8, 16, 32, 64, ...
    const int max_threads = pso.maxTotalThreadsPerThreadgroup;
    const int block_threads = std::min(NxS * warp_size, max_threads);
    const int num_blocks = math::utils::DivUp(NxS, block_threads / warp_size);
    [encoder setComputePipelineState:pso];
    MPSDispatchThreads(num_blocks, block_threads, encoder, pso);
  } else {
    args.pop_back();
    using AccT = typename math::Traits<T>::accumulator_type;
    auto kernel = MPSKernel::TypedString<T>("Block" + name);
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(C, pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:block_warps * sizeof(AccT) atIndex:0];
    MPSDispatchThreads(NxS, block_threads, encoder, pso);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                         \
  template <>                                                   \
  void name<T, MPSContext>(                                     \
      const int N,                                              \
      const int S,                                              \
      const int C,                                              \
      const T* x,                                               \
      T* y,                                                     \
      MPSContext* ctx) {                                        \
    auto* command_buffer = ctx->mps_stream()->command_buffer(); \
    auto* encoder = [command_buffer computeCommandEncoder];     \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];    \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];    \
    DispatchKernel<T>(N, S, C, #name, encoder, ctx);            \
    [encoder endEncoding];                                      \
    [encoder release];                                          \
  }

DEFINE_KERNEL_LAUNCHER(Softmax, float16);
DEFINE_KERNEL_LAUNCHER(Softmax, bfloat16);
DEFINE_KERNEL_LAUNCHER(Softmax, float);
DEFINE_KERNEL_LAUNCHER(Softmax, double);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, bfloat16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                    \
  template <>                                                   \
  void name<T, MPSContext>(                                     \
      const int N,                                              \
      const int S,                                              \
      const int C,                                              \
      const T* dy,                                              \
      const T* y,                                               \
      T* dx,                                                    \
      MPSContext* ctx) {                                        \
    auto* command_buffer = ctx->mps_stream()->command_buffer(); \
    auto* encoder = [command_buffer computeCommandEncoder];     \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:0];   \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];    \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:2];   \
    DispatchKernel<T>(N, S, C, #name, encoder, ctx);            \
    [encoder endEncoding];                                      \
    [encoder release];                                          \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
