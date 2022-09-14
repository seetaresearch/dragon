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

constant uint uint_arg1 [[function_constant(0)]];  // rows | ndim
constant uint uint_arg2 [[function_constant(1)]];  // cols
constant uint4 uint4_arg1 [[function_constant(2)]];
constant uint4 uint4_arg2 [[function_constant(3)]];
constant uint4 uint4_arg3 [[function_constant(4)]];
constant uint4 uint4_arg4 [[function_constant(5)]];
constant SimpleArray<uint, 4, 2> uintarr_arg1 = {uint4_arg1, uint4_arg2};
constant SimpleArray<uint, 4, 2> uintarr_arg2 = {uint4_arg3, uint4_arg4};

template <typename T>
T WarpReduceSum(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val += simd_shuffle_down(val, offset);
  }
  return val;
}

template <typename T, typename AccT>
kernel void RowwiseMoments(
    device const T* x,
    device AccT* mean,
    device AccT* var,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint block_warps = block_size / warp_size;
  threadgroup AccT* m_storage = storage;
  threadgroup AccT* v_storage = storage + block_warps;
  AccT m_val = AccT(0), v_val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const AccT val = AccT(x[j * uint_arg2 + i]);
    m_val += val; v_val += val * val;
  }
  m_val = WarpReduceSum(m_val, warp_size);
  v_val = WarpReduceSum(v_val, warp_size);
  if (lane_id == 0) {
    m_storage[warp_id] = m_val;
    v_storage[warp_id] = v_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (jstart < block_warps) {
    m_val = m_storage[jstart];
    v_val = v_storage[jstart];
  } else {
    m_val = v_val = AccT(0);
  }
  m_val = WarpReduceSum(m_val, warp_size);
  v_val = WarpReduceSum(v_val, warp_size);
  if (jstart == 0) {
    const AccT scale = AccT(1) / AccT(uint_arg1);
    m_val = m_val * scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
kernel void ColwiseMoments(
    device const T* x,
    device AccT* mean,
    device AccT* var,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint block_warps = block_size / warp_size;
  threadgroup AccT* m_storage = storage;
  threadgroup AccT* v_storage = storage + block_warps;
  AccT m_val = AccT(0), v_val = AccT(0);
  for (uint j = jstart; j < uint_arg2; j += block_size) {
    const AccT val = AccT(x[i * uint_arg2 + j]);
    m_val += val; v_val += val * val;
  }
  m_val = WarpReduceSum(m_val, warp_size);
  v_val = WarpReduceSum(v_val, warp_size);
  if (lane_id == 0) {
    m_storage[warp_id] = m_val;
    v_storage[warp_id] = v_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (jstart < block_warps) {
    m_val = m_storage[jstart];
    v_val = v_storage[jstart];
  } else {
    m_val = v_val = AccT(0);
  }
  m_val = WarpReduceSum(m_val, warp_size);
  v_val = WarpReduceSum(v_val, warp_size);
  if (jstart == 0) {
    const AccT scale = AccT(1) / AccT(uint_arg2);
    m_val = m_val * scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
kernel void GenericMoments(
    device const T* x,
    device AccT* mean,
    device AccT* var,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint block_warps = block_size / warp_size;
  threadgroup AccT* m_storage = storage;
  threadgroup AccT* v_storage = storage + block_warps;
  AccT m_val = AccT(0), v_val = AccT(0);
  for (uint j = jstart; j < uint_arg2; j += block_size) {
    uint xi = 0, tmp = i * uint_arg2 + j, r;
    for (int d = uint_arg1 - 1; d >= 0; --d) {
      const int d1 = d / 4, d2 = d % 4;
      FIXED_DIVISOR_DIV_MOD(uintarr_arg1.data[d1][d2], tmp, &tmp, &r);
      xi += r * uintarr_arg2.data[d1][d2];
    }
    const AccT val = AccT(x[xi]);
    m_val += val; v_val += val * val;
  }
  m_val = WarpReduceSum(m_val, warp_size);
  v_val = WarpReduceSum(v_val, warp_size);
  if (lane_id == 0) {
    m_storage[warp_id] = m_val;
    v_storage[warp_id] = v_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (jstart < block_warps) {
    m_val = m_storage[jstart];
    v_val = v_storage[jstart];
  } else {
    m_val = v_val = AccT(0);
  }
  m_val = WarpReduceSum(m_val, warp_size);
  v_val = WarpReduceSum(v_val, warp_size);
  if (jstart == 0) {
    const AccT scale = AccT(1) / AccT(uint_arg2);
    m_val = m_val * scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

#define INSTANTIATE_KERNEL(T, AccT) \
  template [[host_name("RowwiseMoments_"#T)]] \
  kernel void RowwiseMoments(device const T*, device AccT*, device AccT*, \
                             threadgroup AccT*, uint, uint, uint, uint, uint, uint); \
  template [[host_name("ColwiseMoments_"#T)]] \
  kernel void ColwiseMoments(device const T*, device AccT*, device AccT*, \
                             threadgroup AccT*, uint, uint, uint, uint, uint, uint); \
  template [[host_name("GenericMoments_"#T)]] \
  kernel void GenericMoments(device const T*, device AccT*, device AccT*, \
                             threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_KERNEL(half, float);
INSTANTIATE_KERNEL(float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

template <typename T, typename AccT>
void DispatchKernel(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    MPSContext* ctx) {
  int64_t rows, cols;
  vec64_t out_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    out_dims[axes[i]] = 1;
  }
  auto* command_buffer = ctx->mps_stream()->command_buffer();
  auto* encoder = [command_buffer computeCommandEncoder];
  [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];
  [encoder setBuffer:id<MTLBuffer>(mean) offset:0 atIndex:1];
  [encoder setBuffer:id<MTLBuffer>(var) offset:0 atIndex:2];
  MTLComputePipelineState_t pso = nil;
  vector<MPSConstant> args;
  if (math::utils::IsRowwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    auto kernel = MPSKernel::TypedString<T>("RowwiseMoments");
    const uint arg1 = rows, arg2 = cols;
    args.emplace_back(MPSConstant(&arg1, MTLDataTypeUInt, 0));
    args.emplace_back(MPSConstant(&arg2, MTLDataTypeUInt, 1));
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(rows, pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    const auto storage_size = sizeof(AccT) * block_warps * 2;
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:storage_size atIndex:0];
    MPSDispatchThreads(cols, block_threads, encoder, pso);
  } else if (math::utils::IsColwiseReduce(
                 num_dims, dims, out_dims.data(), &rows, &cols)) {
    auto kernel = MPSKernel::TypedString<T>("ColwiseMoments");
    const uint arg1 = rows, arg2 = cols;
    args.emplace_back(MPSConstant(&arg1, MTLDataTypeUInt, 0));
    args.emplace_back(MPSConstant(&arg2, MTLDataTypeUInt, 1));
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(cols, pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    const auto storage_size = sizeof(AccT) * block_warps * 2;
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:storage_size atIndex:0];
    MPSDispatchThreads(rows, block_threads, encoder, pso);
  } else {
    MPS_TENSOR_DIMS_CHECK(num_dims);
    vec64_t transpose_axes(num_dims);
    vec64_t transpose_strides(num_dims);
    vec64_t transpose_dims(num_dims);
    math::utils::TransposeAxesForReduce(
        num_dims, num_axes, axes, transpose_axes.data());
    math::utils::ComputeTransposeStrides(
        num_dims, dims, transpose_axes.data(), transpose_strides.data());
    rows = cols = 1;
    const int pivot = num_dims - num_axes;
    for (int i = 0; i < pivot; ++i) {
      rows *= dims[transpose_axes[i]];
    }
    for (int i = pivot; i < num_dims; ++i) {
      cols *= dims[transpose_axes[i]];
    }
    for (int i = 0; i < num_dims; ++i) {
      transpose_dims[i] = dims[transpose_axes[i]];
    }
    auto kernel = MPSKernel::TypedString<T>("GenericMoments");
    const uint arg1 = num_dims, arg2 = cols;
    vector<uint32_t> arg3(MPS_TENSOR_MAX_DIMS, 0);
    vector<uint32_t> arg4(MPS_TENSOR_MAX_DIMS, 0);
    for (size_t i = 0; i < transpose_dims.size(); ++i) {
      arg3[i] = transpose_dims[i];
    }
    for (size_t i = 0; i < transpose_strides.size(); ++i) {
      arg4[i] = transpose_strides[i];
    }
    args.emplace_back(MPSConstant(&arg1, MTLDataTypeUInt, 0));
    args.emplace_back(MPSConstant(&arg2, MTLDataTypeUInt, 1));
    args.emplace_back(MPSConstant(arg3.data(), MTLDataTypeUInt4, {2, 3}));
    args.emplace_back(MPSConstant(arg4.data(), MTLDataTypeUInt4, {4, 5}));
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(cols, pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    const auto storage_size = sizeof(AccT) * block_warps * 2;
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:storage_size atIndex:0];
    MPSDispatchThreads(rows, block_threads, encoder, pso);
  }
  [encoder endEncoding];
  [encoder release];
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                      \
  template <>                                                \
  void Moments<T, AccT, MPSContext>(                         \
      const int num_dims,                                    \
      const int64_t* dims,                                   \
      const int num_axes,                                    \
      const int64_t* axes,                                   \
      const T* x,                                            \
      AccT* mean,                                            \
      AccT* var,                                             \
      MPSContext* ctx) {                                     \
    vec64_t new_dims, new_axes;                              \
    math::utils::CollapseReduceAxes(                         \
        num_dims, dims, num_axes, axes, new_dims, new_axes); \
    DispatchKernel<T, AccT>(                                 \
        new_dims.size(),                                     \
        new_dims.data(),                                     \
        new_axes.size(),                                     \
        new_axes.data(),                                     \
        x,                                                   \
        mean,                                                \
        var,                                                 \
        ctx);                                                \
  }

DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
