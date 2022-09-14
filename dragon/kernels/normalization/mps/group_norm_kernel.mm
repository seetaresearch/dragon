#include "dragon/kernels/normalization/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // G
constant uint uint_arg2 [[function_constant(1)]]; // D
constant uint uint_arg3 [[function_constant(2)]]; // S
constant uint uint_arg4 [[function_constant(3)]]; // NxS

typedef enum {
  NCHW = 0,
  NHWC = 1,
} StorageOrder;

template <typename T>
T Cube(const T val) {
  return val * val * val;
}

template <typename T>
T WarpReduceSum(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val += simd_shuffle_down(val, offset);
  }
  return val;
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void GroupNorm(
    device const T* x,
    device const AccT* mu,
    device const AccT* rsig,
    device const AccT* gamma,
    device const AccT* beta,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const uint C = uint_arg1 * uint_arg2;
  const int ng = kOrder == StorageOrder::NCHW ?
    index / (uint_arg2 * uint_arg3) :
    index / (C * uint_arg3) * uint_arg1 + (index / uint_arg2 % uint_arg1);
  const int c = kOrder == StorageOrder::NCHW ? index / uint_arg3 % C : index % C;
  y[index] = T(fma((AccT(x[index]) - mu[ng]) * rsig[ng], gamma[c], beta[c]));
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void GroupNormWGrad(
    device const T* x,
    device const AccT* mu,
    device const AccT* rsig,
    device const T* dy,
    device AccT* dgamma,
    device AccT* dbeta,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint C = uint_arg1 * uint_arg2;
  const uint block_warps = block_size / warp_size;
  threadgroup AccT* dg_storage = storage;
  threadgroup AccT* db_storage = storage + block_warps;
  AccT dg_val = AccT(0), db_val = AccT(0);
  for (uint j = jstart; j < uint_arg4; j += block_size) {
    const uint n = j / uint_arg3;
    const uint ng = n * uint_arg1 + i / uint_arg2;
    const uint index = kOrder == StorageOrder::NCHW ?
      (n * C + i) * uint_arg3 + j % uint_arg3 :
      j * C + i;
    dg_val += AccT(dy[index]) * (AccT(x[index]) - mu[ng]) * rsig[ng];
    db_val += AccT(dy[index]);
  }
  dg_val = WarpReduceSum(dg_val, warp_size);
  db_val = WarpReduceSum(db_val, warp_size);
  if (lane_id == 0) {
    dg_storage[warp_id] = dg_val;
    db_storage[warp_id] = db_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (jstart < block_warps) {
    dg_val = dg_storage[jstart];
    db_val = db_storage[jstart];
  } else {
    dg_val = db_val = AccT(0);
  }
  dg_val = WarpReduceSum(dg_val, warp_size);
  db_val = WarpReduceSum(db_val, warp_size);
  if (jstart == 0) {
    dgamma[i] = dg_val;
    dbeta[i] = db_val;
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void GroupNormInternalGrad(
    device const T* x,
    device const AccT* gamma,
    device const T* dy,
    device AccT* ds,
    device AccT* db,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint C = uint_arg1 * uint_arg2;
  const uint DxS = uint_arg2 * uint_arg3;
  const uint block_warps = block_size / warp_size;
  threadgroup AccT* ds_storage = storage;
  threadgroup AccT* db_storage = storage + block_warps;
  AccT ds_val = AccT(0), db_val = AccT(0);
  for (uint j = jstart; j < DxS; j += block_size) {
    const int c = i % uint_arg1 * uint_arg2 + j / uint_arg3;
    const int index = kOrder == StorageOrder::NCHW ?
      i * DxS + j :
      (i / uint_arg1 * uint_arg3 + j % uint_arg3) * C + c;
    const AccT val = gamma[c] * AccT(dy[index]);
    ds_val += val * AccT(x[index]);
    db_val += val;
  }
  ds_val = WarpReduceSum(ds_val, warp_size);
  db_val = WarpReduceSum(db_val, warp_size);
  if (lane_id == 0) {
    ds_storage[warp_id] = ds_val;
    db_storage[warp_id] = db_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (jstart < block_warps) {
    ds_val = ds_storage[jstart];
    db_val = db_storage[jstart];
  } else {
    ds_val = db_val = AccT(0);
  }
  ds_val = WarpReduceSum(ds_val, warp_size);
  db_val = WarpReduceSum(db_val, warp_size);  
  if (jstart == 0) {
    ds[i] = ds_val;
    db[i] = db_val;
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void GroupNormGrad(
    device const T* x,
    device const AccT* mu,
    device const AccT* rsig,
    device const AccT* gamma,
    device const AccT* ds,
    device const AccT* db,
    device const T* dy,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  const uint C = uint_arg1 * uint_arg2;
  const uint DxS = uint_arg2 * uint_arg3;
  const AccT denom = AccT(1) / AccT(DxS);
  const uint ng = kOrder == StorageOrder::NCHW ? 
    index / DxS :
    index / (C * uint_arg3) * uint_arg1 + (index / uint_arg2 % uint_arg1);
  const uint c = kOrder == StorageOrder::NCHW ? index / uint_arg3 % C : index % C;
  const AccT u = fma(db[ng], mu[ng], -ds[ng]) * (AccT(x[index]) - mu[ng]) * Cube(rsig[ng]);
  const AccT v = db[ng] * rsig[ng];
  dx[index] = T(gamma[c] * AccT(dy[index]) * rsig[ng] + (u - v) * denom);
}


#define INSTANTIATE_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
      device const T*, device const AccT*, device const AccT*, \
      device const AccT*, device const AccT*, device T*, uint);

INSTANTIATE_KERNEL(GroupNorm, NCHW, half, float);
INSTANTIATE_KERNEL(GroupNorm, NCHW, float, float);
INSTANTIATE_KERNEL(GroupNorm, NHWC, half, float);
INSTANTIATE_KERNEL(GroupNorm, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(GroupNorm, NCHW, double, double);
INSTANTIATE_KERNEL(GroupNorm, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
     device const T*, device const AccT*, device const AccT*, \
     device const T*, device AccT*, device AccT*, \
     threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_GRAD_KERNEL(GroupNormWGrad, NCHW, half, float);
INSTANTIATE_GRAD_KERNEL(GroupNormWGrad, NCHW, float, float);
INSTANTIATE_GRAD_KERNEL(GroupNormWGrad, NHWC, half, float);
INSTANTIATE_GRAD_KERNEL(GroupNormWGrad, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(GroupNormWGrad, NCHW, double, double);
INSTANTIATE_GRAD_KERNEL(GroupNormWGrad, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
     device const T*, device const AccT*, device const T*, \
     device AccT*, device AccT*, \
     threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_GRAD_KERNEL(GroupNormInternalGrad, NCHW, half, float);
INSTANTIATE_GRAD_KERNEL(GroupNormInternalGrad, NCHW, float, float);
INSTANTIATE_GRAD_KERNEL(GroupNormInternalGrad, NHWC, half, float);
INSTANTIATE_GRAD_KERNEL(GroupNormInternalGrad, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(GroupNormInternalGrad, NCHW, double, double);
INSTANTIATE_GRAD_KERNEL(GroupNormInternalGrad, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
     device const T*, device const AccT*, device const AccT*, \ 
     device const AccT*, device const AccT*, device const AccT*, \
     device const T*, device T*, uint);

INSTANTIATE_GRAD_KERNEL(GroupNormGrad, NCHW, half, float);
INSTANTIATE_GRAD_KERNEL(GroupNormGrad, NCHW, float, float);
INSTANTIATE_GRAD_KERNEL(GroupNormGrad, NHWC, half, float);
INSTANTIATE_GRAD_KERNEL(GroupNormGrad, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(GroupNormGrad, NCHW, double, double);
INSTANTIATE_GRAD_KERNEL(GroupNormGrad, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                                 \
  template <>                                                           \
  void GroupNorm<T, AccT, MPSContext>(                                  \
      const int N,                                                      \
      const int G,                                                      \
      const int D,                                                      \
      const int S,                                                      \
      const string& data_format,                                        \
      const T* x,                                                       \
      const AccT* mu,                                                   \
      const AccT* rsig,                                                 \
      const AccT* gamma,                                                \
      const AccT* beta,                                                 \
      T* y,                                                             \
      MPSContext* ctx) {                                                \
    auto kernel = MPSKernel::TypedString<T>("GroupNorm" + data_format); \
    auto* command_buffer = ctx->mps_stream()->command_buffer();         \
    auto* encoder = [command_buffer computeCommandEncoder];             \
    const uint arg1 = G, arg2 = D, arg3 = S;                            \
    auto args = vector<MPSConstant>({                                   \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                         \
        MPSConstant(&arg2, MTLDataTypeUInt, 1),                         \
        MPSConstant(&arg3, MTLDataTypeUInt, 2),                         \
    });                                                                 \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);   \
    [encoder setComputePipelineState:pso];                              \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];            \
    [encoder setBuffer:id<MTLBuffer>(mu) offset:0 atIndex:1];           \
    [encoder setBuffer:id<MTLBuffer>(rsig) offset:0 atIndex:2];         \
    [encoder setBuffer:id<MTLBuffer>(gamma) offset:0 atIndex:3];        \
    [encoder setBuffer:id<MTLBuffer>(beta) offset:0 atIndex:4];         \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:5];            \
    MPSDispatchThreads((N * G * D * S), encoder, pso);                  \
    [encoder endEncoding];                                              \
    [encoder release];                                                  \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                                   \
  template <>                                                                  \
  void GroupNormGrad<T, AccT, MPSContext>(                                     \
      const int N,                                                             \
      const int G,                                                             \
      const int D,                                                             \
      const int S,                                                             \
      const string& data_format,                                               \
      const T* x,                                                              \
      const AccT* mu,                                                          \
      const AccT* rsig,                                                        \
      const AccT* gamma,                                                       \
      const T* dy,                                                             \
      AccT* ds,                                                                \
      AccT* db,                                                                \
      AccT* dgamma,                                                            \
      AccT* dbeta,                                                             \
      T* dx,                                                                   \
      MPSContext* ctx) {                                                       \
    size_t db_offset = ds == db ? sizeof(AccT) * (N * G) : 0;                  \
    auto* command_buffer = ctx->mps_stream()->command_buffer();                \
    const uint arg1 = G, arg2 = D, arg3 = S, arg4 = N * S;                     \
    auto args = vector<MPSConstant>({                                          \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                                \
        MPSConstant(&arg2, MTLDataTypeUInt, 1),                                \
        MPSConstant(&arg3, MTLDataTypeUInt, 2),                                \
        MPSConstant(&arg4, MTLDataTypeUInt, 3),                                \
    });                                                                        \
    {                                                                          \
      auto kernel = MPSKernel::TypedString<T>("GroupNormWGrad" + data_format); \
      auto* encoder = [command_buffer computeCommandEncoder];                  \
      auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);        \
      const int block_threads = MPSGetBlockReduceThreads(N * S, pso);          \
      const int block_warps = block_threads / pso.threadExecutionWidth;        \
      const auto storage_size = sizeof(AccT) * block_warps * 2;                \
      [encoder setComputePipelineState:pso];                                   \
      [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                 \
      [encoder setBuffer:id<MTLBuffer>(mu) offset:0 atIndex:1];                \
      [encoder setBuffer:id<MTLBuffer>(rsig) offset:0 atIndex:2];              \
      [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:3];                \
      [encoder setBuffer:id<MTLBuffer>(dgamma) offset:0 atIndex:4];            \
      [encoder setBuffer:id<MTLBuffer>(dbeta) offset:0 atIndex:5];             \
      [encoder setThreadgroupMemoryLength:storage_size atIndex:0];             \
      MPSDispatchThreads((G * D), block_threads, encoder, pso);                \
      [encoder endEncoding];                                                   \
      [encoder release];                                                       \
    }                                                                          \
    args.pop_back();                                                           \
    {                                                                          \
      auto kernel = "GroupNormInternalGrad" + data_format;                     \
      kernel = MPSKernel::TypedString<T>(kernel);                              \
      auto* encoder = [command_buffer computeCommandEncoder];                  \
      auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);        \
      const int block_threads = MPSGetBlockReduceThreads(D * S, pso);          \
      const int block_warps = block_threads / pso.threadExecutionWidth;        \
      const auto storage_size = sizeof(AccT) * block_warps * 2;                \
      [encoder setComputePipelineState:pso];                                   \
      [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                 \
      [encoder setBuffer:id<MTLBuffer>(gamma) offset:0 atIndex:1];             \
      [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:2];                \
      [encoder setBuffer:id<MTLBuffer>(ds) offset:0 atIndex:3];                \
      [encoder setBuffer:id<MTLBuffer>(db) offset:db_offset atIndex:4];        \
      [encoder setThreadgroupMemoryLength:storage_size atIndex:0];             \
      MPSDispatchThreads((N * G), block_threads, encoder, pso);                \
      [encoder endEncoding];                                                   \
      [encoder release];                                                       \
    }                                                                          \
    {                                                                          \
      auto kernel = MPSKernel::TypedString<T>("GroupNormGrad" + data_format);  \
      auto* encoder = [command_buffer computeCommandEncoder];                  \
      auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);        \
      [encoder setComputePipelineState:pso];                                   \
      [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                 \
      [encoder setBuffer:id<MTLBuffer>(mu) offset:0 atIndex:1];                \
      [encoder setBuffer:id<MTLBuffer>(rsig) offset:0 atIndex:2];              \
      [encoder setBuffer:id<MTLBuffer>(gamma) offset:0 atIndex:3];             \
      [encoder setBuffer:id<MTLBuffer>(ds) offset:0 atIndex:4];                \
      [encoder setBuffer:id<MTLBuffer>(db) offset:db_offset atIndex:5];        \
      [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:6];                \
      [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:7];                \
      MPSDispatchThreads((N * G * D * S), encoder, pso);                       \
      [encoder endEncoding];                                                   \
      [encoder release];                                                       \
    }                                                                          \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
