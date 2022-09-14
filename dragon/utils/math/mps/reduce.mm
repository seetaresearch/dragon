#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

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

typedef enum {
  Add = 0,
  Min = 1,
  Max = 2,
} ReduceOp;

constant float float_arg1 [[function_constant(0)]]; // init
constant float float_arg2 [[function_constant(1)]]; // scale
constant uint uint_arg1 [[function_constant(2)]];   // rows | ndim
constant uint uint_arg2 [[function_constant(3)]];   // cols
constant uint4 uint4_arg1 [[function_constant(4)]];
constant uint4 uint4_arg2 [[function_constant(5)]];
constant uint4 uint4_arg3 [[function_constant(6)]];
constant uint4 uint4_arg4 [[function_constant(7)]];
constant SimpleArray<uint, 4, 2> uintarr_arg1 = {uint4_arg1, uint4_arg2};
constant SimpleArray<uint, 4, 2> uintarr_arg2 = {uint4_arg3, uint4_arg4};

template <typename T>
T ReduceFunc(const T lhs, const T rhs, ReduceOp op) {
  if (op == Add) return lhs + rhs;
  else if (op == Min) return min(lhs, rhs);
  else if (op == Max) return max(lhs, rhs);
  return lhs;
}

template <typename T>
T WarpReduce(T val, ReduceOp op, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    if (op == Add) val += simd_shuffle_down(val, offset);
    else if (op == Min) val = min(val, simd_shuffle_down(val, offset));
    else if (op == Max) val = max(val, simd_shuffle_down(val, offset));
  }
  return val;
}

template <typename T, typename AccT>
void DeviceReduce(
    ReduceOp op,
    device const T* x,
    device _atomic<T> *y,
    threadgroup AccT* storage,
    const uint block_size,
    const uint warp_size,
    const uint index,
    const uint thread_id,
    const uint lane_id,
    const uint warp_id) {
  AccT val = AccT(float_arg1);
  if (index < uint_arg1) val = ReduceFunc(val, AccT(x[index]), op);
  val = WarpReduce(val, op, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = thread_id < (block_size / warp_size) ? storage[thread_id] : AccT(float_arg1);
  val = WarpReduce(val, op, warp_size);
  if (thread_id == 0) {
    atomic_fetch_add_explicit(y, T(val * AccT(float_arg2)), memory_order_relaxed);
  }
}

#define DEFINE_REDUCE_KERNEL(impl, name, op)                   \
  template <typename T, typename AccT>                         \
  kernel void impl##name(                                      \
      device const T* x,                                       \
      device _atomic<T>* y,                                    \
      threadgroup AccT* storage [[threadgroup(0)]],            \
      const uint block_size [[threads_per_threadgroup]],       \
      const uint warp_size [[threads_per_simdgroup]],          \
      const uint index [[thread_position_in_grid]],            \
      const uint thread_id [[thread_position_in_threadgroup]], \
      const uint lane_id [[thread_index_in_simdgroup]],        \
      const uint warp_id [[simdgroup_index_in_threadgroup]]) { \
    impl(op, x, y, storage, block_size, warp_size,             \
         index, thread_id, lane_id, warp_id);                  \
  }

DEFINE_REDUCE_KERNEL(DeviceReduce, Max, Max);
DEFINE_REDUCE_KERNEL(DeviceReduce, Min, Min);
DEFINE_REDUCE_KERNEL(DeviceReduce, Sum, Add);
#undef DEFINE_REDUCE_KERNEL

template <typename T, typename AccT>
void RowwiseReduce(
    ReduceOp op,
    device const T* x,
    device T* y,
    threadgroup AccT* storage,
    const uint block_size,
    const uint warp_size,
    const uint i,
    const uint jstart,
    const uint lane_id,
    const uint warp_id) {
  AccT val = AccT(float_arg1);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    val = ReduceFunc(val, AccT(x[j * uint_arg2 + i]), op);
  }
  val = WarpReduce(val, op, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < (block_size / warp_size) ? storage[jstart] : AccT(float_arg1);
  val = WarpReduce(val, op, warp_size);
  if (jstart == 0) y[i] = T(val * AccT(float_arg2));
}

template <typename T, typename AccT>
void ColwiseReduce(
    ReduceOp op,
    device const T* x,
    device T* y,
    threadgroup AccT* storage,
    const uint block_size,
    const uint warp_size,
    const uint i,
    const uint jstart,
    const uint lane_id,
    const uint warp_id) {
  AccT val = AccT(float_arg1);
  for (uint j = jstart; j < uint_arg2; j += block_size) {
    val = ReduceFunc(val, AccT(x[i * uint_arg2 + j]), op);
  }
  val = WarpReduce(val, op, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < (block_size / warp_size) ? storage[jstart] : AccT(float_arg1);
  val = WarpReduce(val, op, warp_size);
  if (jstart == 0) y[i] = T(val * AccT(float_arg2));
}

template <typename T, typename AccT>
void GenericReduce(
    ReduceOp op,
    device const T* x,
    device T* y,
    threadgroup AccT* storage,
    const uint block_size,
    const uint warp_size,
    const uint i,
    const uint jstart,
    const uint lane_id,
    const uint warp_id) {
  AccT val = AccT(float_arg1);
  for (uint j = jstart; j < uint_arg2; j += block_size) {
    uint xi = 0, tmp = i * uint_arg2 + j, r;
    for (int d = uint_arg1 - 1; d >= 0; --d) {
      const int d1 = d / 4, d2 = d % 4;
      FIXED_DIVISOR_DIV_MOD(uintarr_arg1.data[d1][d2], tmp, &tmp, &r);
      xi += r * uintarr_arg2.data[d1][d2];
    }
    val = ReduceFunc(val, AccT(x[xi]), op);
  }
  val = WarpReduce(val, op, warp_size);
  if (lane_id == 0) storage[warp_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  val = jstart < (block_size / warp_size) ? storage[jstart] : AccT(float_arg1);
  val = WarpReduce(val, op, warp_size);
  if (jstart == 0) y[i] = T(val * AccT(float_arg2));
}

#define DEFINE_REDUCE_KERNEL(impl, name, op) \
  template <typename T, typename AccT> \
  kernel void impl##name( \
      device const T* x, \
      device T* y, \
      threadgroup AccT* storage [[threadgroup(0)]], \
      const uint block_size [[threads_per_threadgroup]], \
      const uint warp_size [[threads_per_simdgroup]], \
      const uint i [[threadgroup_position_in_grid]], \
      const uint jstart [[thread_position_in_threadgroup]], \
      const uint lane_id [[thread_index_in_simdgroup]], \
      const uint warp_id [[simdgroup_index_in_threadgroup]]) { \
    impl(op, x, y, storage, block_size, warp_size, i, jstart, lane_id, warp_id); \
  }

DEFINE_REDUCE_KERNEL(RowwiseReduce, Sum, Add);
DEFINE_REDUCE_KERNEL(RowwiseReduce, Min, Min);
DEFINE_REDUCE_KERNEL(RowwiseReduce, Max, Max);
DEFINE_REDUCE_KERNEL(ColwiseReduce, Sum, Add);
DEFINE_REDUCE_KERNEL(ColwiseReduce, Min, Min);
DEFINE_REDUCE_KERNEL(ColwiseReduce, Max, Max);
DEFINE_REDUCE_KERNEL(GenericReduce, Sum, Add);
DEFINE_REDUCE_KERNEL(GenericReduce, Min, Min);
DEFINE_REDUCE_KERNEL(GenericReduce, Max, Max);
#undef DEFINE_REDUCE_KERNEL

#define INSTANTIATE_REDUCE_KERNEL(name, T, AccT) \
  template [[host_name("RowwiseReduce"#name"_"#T)]] \
  kernel void RowwiseReduce##name(device const T*, device T*, threadgroup AccT*, \
                                  uint, uint, uint, uint, uint, uint); \
  template [[host_name("ColwiseReduce"#name"_"#T)]] \
  kernel void ColwiseReduce##name(device const T*, device T*, threadgroup AccT*, \
                                  uint, uint, uint, uint, uint, uint); \
  template [[host_name("GenericReduce"#name"_"#T)]] \
  kernel void GenericReduce##name(device const T*, device T*, threadgroup AccT*, \
                                  uint, uint, uint, uint, uint, uint);

INSTANTIATE_REDUCE_KERNEL(Max, uint8_t, uint8_t);
INSTANTIATE_REDUCE_KERNEL(Max, int8_t, int8_t);
INSTANTIATE_REDUCE_KERNEL(Max, int, int);
INSTANTIATE_REDUCE_KERNEL(Max, int64_t, int); // Fallback.
INSTANTIATE_REDUCE_KERNEL(Max, half, float);
INSTANTIATE_REDUCE_KERNEL(Max, float, float);
INSTANTIATE_REDUCE_KERNEL(Min, uint8_t, uint8_t);
INSTANTIATE_REDUCE_KERNEL(Min, int8_t, int8_t);
INSTANTIATE_REDUCE_KERNEL(Min, int, int);
INSTANTIATE_REDUCE_KERNEL(Min, int64_t, int); // Fallback.
INSTANTIATE_REDUCE_KERNEL(Min, half, float);
INSTANTIATE_REDUCE_KERNEL(Min, float, float);
INSTANTIATE_REDUCE_KERNEL(Sum, int, int);
INSTANTIATE_REDUCE_KERNEL(Sum, int64_t, int); // Fallback.
INSTANTIATE_REDUCE_KERNEL(Sum, half, float);
INSTANTIATE_REDUCE_KERNEL(Sum, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_REDUCE_KERNEL(Max, double, double);
INSTANTIATE_REDUCE_KERNEL(Min, double, double);
INSTANTIATE_REDUCE_KERNEL(Sum, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_REDUCE_KERNEL

#define INSTANTIATE_REDUCE_KERNEL(name, T, AccT) \
  template [[host_name("DeviceReduce"#name"_"#T)]] \
  kernel void DeviceReduce##name( \
      device const T*, device _atomic<T>*, threadgroup AccT*, \
      uint, uint, uint, uint, uint, uint);

INSTANTIATE_REDUCE_KERNEL(Max, int, int);
INSTANTIATE_REDUCE_KERNEL(Min, int, int);
INSTANTIATE_REDUCE_KERNEL(Sum, int, int);
#undef INSTANTIATE_REDUCE_KERNEL

)";

template <typename T, typename AccT>
void DispatchReduce(
    const string& op,
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const float init,
    const float scale,
    const T* x,
    T* y,
    MPSContext* ctx) {
  int64_t rows, cols;
  vec64_t out_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    out_dims[axes[i]] = 1;
  }
  const auto meta = TypeMeta::Make<T>();
  bool HasDeviceReduce = meta.template Match<int>();
  auto args = vector<MPSConstant>({
      MPSConstant(&init, MTLDataTypeFloat, 0),
      MPSConstant(&scale, MTLDataTypeFloat, 1),
  });
  auto* command_buffer = ctx->mps_stream()->command_buffer();
  if (num_dims == num_axes && HasDeviceReduce) ctx->MemsetAsync(sizeof(T), y);
  auto* encoder = [command_buffer computeCommandEncoder];
  [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];
  [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:1];
  MTLComputePipelineState_t pso = nil;
  if (num_dims == num_axes && HasDeviceReduce) {
    auto kernel = MPSKernel::TypedString<T>("DeviceReduce" + op);
    const uint arg3 = math::utils::Prod(num_dims, dims);
    args.emplace_back(MPSConstant(&arg3, MTLDataTypeUInt, 2));
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(int(arg3), pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    const int num_blocks = math::utils::DivUp(int(arg3), block_threads);
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:block_warps * sizeof(AccT) atIndex:0];
    MPSDispatchThreads(num_blocks, block_threads, encoder, pso);
  } else if (math::utils::IsRowwiseReduce(
                 num_dims, dims, out_dims.data(), &rows, &cols)) {
    auto kernel = MPSKernel::TypedString<T>("RowwiseReduce" + op);
    const uint arg3 = rows, arg4 = cols;
    args.emplace_back(MPSConstant(&arg3, MTLDataTypeUInt, 2));
    args.emplace_back(MPSConstant(&arg4, MTLDataTypeUInt, 3));
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(rows, pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:block_warps * sizeof(AccT) atIndex:0];
    MPSDispatchThreads(cols, block_threads, encoder, pso);
  } else if (math::utils::IsColwiseReduce(
                 num_dims, dims, out_dims.data(), &rows, &cols)) {
    auto kernel = MPSKernel::TypedString<T>("ColwiseReduce" + op);
    const uint arg3 = rows, arg4 = cols;
    args.emplace_back(MPSConstant(&arg3, MTLDataTypeUInt, 2));
    args.emplace_back(MPSConstant(&arg4, MTLDataTypeUInt, 3));
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(cols, pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:block_warps * sizeof(AccT) atIndex:0];
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
    auto kernel = MPSKernel::TypedString<T>("GenericReduce" + op);
    const uint arg3 = num_dims, arg4 = cols;
    vector<uint32_t> arg5(MPS_TENSOR_MAX_DIMS, 0);
    vector<uint32_t> arg6(MPS_TENSOR_MAX_DIMS, 0);
    for (size_t i = 0; i < transpose_dims.size(); ++i) {
      arg5[i] = transpose_dims[i];
    }
    for (size_t i = 0; i < transpose_strides.size(); ++i) {
      arg6[i] = transpose_strides[i];
    }
    args.emplace_back(MPSConstant(&arg3, MTLDataTypeUInt, 2));
    args.emplace_back(MPSConstant(&arg4, MTLDataTypeUInt, 3));
    args.emplace_back(MPSConstant(arg5.data(), MTLDataTypeUInt4, {4, 5}));
    args.emplace_back(MPSConstant(arg6.data(), MTLDataTypeUInt4, {6, 7}));
    pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
    const int block_threads = MPSGetBlockReduceThreads(cols, pso);
    const int block_warps = block_threads / pso.threadExecutionWidth;
    [encoder setComputePipelineState:pso];
    [encoder setThreadgroupMemoryLength:block_warps * sizeof(AccT) atIndex:0];
    MPSDispatchThreads(rows, block_threads, encoder, pso);
  }
  [encoder endEncoding];
  [encoder release];
}

} // namespace

#define DEFINE_REDUCE_FUNC(name, T, AccT, kInit)             \
  template <>                                                \
  DRAGON_API void Reduce##name<T, MPSContext>(               \
      const int num_dims,                                    \
      const int64_t* dims,                                   \
      const int num_axes,                                    \
      const int64_t* axes,                                   \
      const float scale,                                     \
      const T* x,                                            \
      T* y,                                                  \
      MPSContext* ctx) {                                     \
    vec64_t new_dims, new_axes;                              \
    math::utils::CollapseReduceAxes(                         \
        num_dims, dims, num_axes, axes, new_dims, new_axes); \
    DispatchReduce<T, AccT>(                                 \
        #name,                                               \
        new_dims.size(),                                     \
        new_dims.data(),                                     \
        new_axes.size(),                                     \
        new_axes.data(),                                     \
        float(kInit),                                        \
        float(scale),                                        \
        x,                                                   \
        y,                                                   \
        ctx);                                                \
  }

DEFINE_REDUCE_FUNC(
    Max,
    uint8_t,
    uint8_t,
    std::numeric_limits<uint8_t>::lowest());
DEFINE_REDUCE_FUNC(Max, int8_t, int8_t, std::numeric_limits<int8_t>::lowest());
DEFINE_REDUCE_FUNC(Max, int, int, std::numeric_limits<int>::lowest());
DEFINE_REDUCE_FUNC(
    Max,
    int64_t,
    int64_t,
    std::numeric_limits<int64_t>::lowest());
DEFINE_REDUCE_FUNC(Max, float16, float, -65505.f);
DEFINE_REDUCE_FUNC(Max, float, float, std::numeric_limits<float>::lowest());
DEFINE_REDUCE_FUNC(Max, double, double, std::numeric_limits<double>::lowest());
DEFINE_REDUCE_FUNC(Min, uint8_t, uint8_t, std::numeric_limits<uint8_t>::max());
DEFINE_REDUCE_FUNC(Min, int8_t, int8_t, std::numeric_limits<int8_t>::max());
DEFINE_REDUCE_FUNC(Min, int, int, std::numeric_limits<int>::max());
DEFINE_REDUCE_FUNC(Min, int64_t, int64_t, std::numeric_limits<int64_t>::max());
DEFINE_REDUCE_FUNC(Min, float16, float, 65504.f);
DEFINE_REDUCE_FUNC(Min, float, float, std::numeric_limits<float>::max());
DEFINE_REDUCE_FUNC(Min, double, double, std::numeric_limits<double>::max());
DEFINE_REDUCE_FUNC(Sum, int, int, int(0));
DEFINE_REDUCE_FUNC(Sum, int64_t, int64_t, int64_t(0));
DEFINE_REDUCE_FUNC(Sum, float16, float, 0.f);
DEFINE_REDUCE_FUNC(Sum, float, float, 0.f);
DEFINE_REDUCE_FUNC(Sum, double, double, 0.);
#undef DEFINE_REDUCE_FUNC

#define DEFINE_SUM_FUNC(T)                                                 \
  template <>                                                              \
  DRAGON_API void Sum<T, MPSContext>(                                      \
      const int N, const float alpha, const T* x, T* y, MPSContext* ctx) { \
    vec64_t dims = {N}, axes = {0};                                        \
    math::ReduceSum(1, dims.data(), 1, axes.data(), alpha, x, y, ctx);     \
  }

DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float16);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

} // namespace math

} // namespace dragon
