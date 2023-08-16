#include "dragon/kernels/normalization/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]]; // C
constant uint uint_arg2 [[function_constant(1)]]; // S
constant uint uint_arg3 [[function_constant(2)]]; // NxS
constant float float_arg1 [[function_constant(3)]]; // normalizer

typedef enum {
  NCHW = 0,
  NHWC = 1,
} StorageOrder;

template <typename T>
T WarpReduceSum(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val += simd_shuffle_down(val, offset);
  }
  return val;
}

template <typename T>
kernel void BatchNormFuseParams(
    device const T* mu,
    device const T* rsig,
    device const T* gamma,
    device const T* beta,
    device T* scale,
    device T* bias,
    const uint index [[thread_position_in_grid]]) {
  const T val = scale[index] = gamma[index] * rsig[index];
  bias[index] = fma(-val, mu[index], beta[index]);
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void BatchNormAffine(
    device const T* x,
    device const AccT* scale,
    device const AccT* bias,
    device T* y,
    const uint index [[thread_position_in_grid]]) {
  const uint j = kOrder == NCHW ? (index / uint_arg2) % uint_arg1
                                : index % uint_arg1;
  y[index] = T(fma(AccT(x[index]), scale[j], bias[j]));
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void BatchNormWGrad(
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
  const uint block_warps = block_size / warp_size;
  threadgroup AccT* dg_storage = storage;
  threadgroup AccT* db_storage = storage + block_warps;
  AccT dg_val = AccT(0), db_val = AccT(0);
  for (uint j = jstart; j < uint_arg3; j += block_size) {
    const uint index = kOrder == NCHW ?
      (j / uint_arg2 * uint_arg1 + i) * uint_arg2 + j % uint_arg2 :
      j * uint_arg1 + i;
    dg_val += dy[index] * (x[index] - mu[i]);
    db_val +=  dy[index];
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
    dgamma[i] = dg_val * rsig[i];
    dbeta[i] = db_val;
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void BatchNormTrainingGrad(
    device const T* x,
    device const AccT* mu,
    device const AccT* rsig,
    device const AccT* gamma,
    device const AccT* dgamma,
    device const AccT* dbeta,
    device const T* dy,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  const int i = kOrder == StorageOrder::NCHW ? index / uint_arg2 % uint_arg1
                                             : index % uint_arg1;
  dx[index] = T(gamma[i] * rsig[i] * (
    AccT(dy[index]) - fma((AccT(x[index]) - mu[i]) * rsig[i],
                          dgamma[i], dbeta[i]) / AccT(float_arg1)));
}

template <typename T, typename AccT, StorageOrder kOrder>
kernel void BatchNormInferenceGrad(
    device const AccT* rsig,
    device const AccT* gamma,
    device const T* dy,
    device T* dx,
    const uint index [[thread_position_in_grid]]) {
  const int i = kOrder == StorageOrder::NCHW ? index / uint_arg2 % uint_arg1
                                             : index % uint_arg1;
  dx[index] = T(gamma[i] * AccT(dy[index]) * rsig[i]);
}

#define INSTANTIATE_KERNEL(name, T) \
  template [[host_name(#name"_"#T)]] \
  kernel void name(device const T*, device const T*, \
                   device const T*, device const T*, \
                   device T*, device T*, uint);

INSTANTIATE_KERNEL(BatchNormFuseParams, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(BatchNormFuseParams, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
      device const T*, device const AccT*, device const AccT*, \
      device T*, uint);

INSTANTIATE_KERNEL(BatchNormAffine, NCHW, half, float);
INSTANTIATE_KERNEL(BatchNormAffine, NCHW, float, float);
INSTANTIATE_KERNEL(BatchNormAffine, NHWC, half, float);
INSTANTIATE_KERNEL(BatchNormAffine, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(BatchNormAffine, NCHW, double, double);
INSTANTIATE_KERNEL(BatchNormAffine, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
     device const T*, device const AccT*, device const AccT*, \
     device const T*, device AccT*, device AccT*, \
     threadgroup AccT*, uint, uint, uint, uint, uint, uint);

INSTANTIATE_GRAD_KERNEL(BatchNormWGrad, NCHW, half, float);
INSTANTIATE_GRAD_KERNEL(BatchNormWGrad, NCHW, float, float);
INSTANTIATE_GRAD_KERNEL(BatchNormWGrad, NHWC, half, float);
INSTANTIATE_GRAD_KERNEL(BatchNormWGrad, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(BatchNormWGrad, NCHW, double, double);
INSTANTIATE_GRAD_KERNEL(BatchNormWGrad, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
     device const T*, device const AccT*, device const AccT*, \
     device const AccT*, device const AccT*, device const AccT*, \
     device const T*, device T*, uint);

INSTANTIATE_GRAD_KERNEL(BatchNormTrainingGrad, NCHW, half, float);
INSTANTIATE_GRAD_KERNEL(BatchNormTrainingGrad, NCHW, float, float);
INSTANTIATE_GRAD_KERNEL(BatchNormTrainingGrad, NHWC, half, float);
INSTANTIATE_GRAD_KERNEL(BatchNormTrainingGrad, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(BatchNormTrainingGrad, NCHW, double, double);
INSTANTIATE_GRAD_KERNEL(BatchNormTrainingGrad, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

#define INSTANTIATE_GRAD_KERNEL(name, kOrder, T, AccT) \
  template [[host_name(#name#kOrder"_"#T)]] \
  kernel void name<T, AccT, kOrder>( \
     device const AccT*, device const AccT*, \
     device const T*, device T*, uint);

INSTANTIATE_GRAD_KERNEL(BatchNormInferenceGrad, NCHW, half, float);
INSTANTIATE_GRAD_KERNEL(BatchNormInferenceGrad, NCHW, float, float);
INSTANTIATE_GRAD_KERNEL(BatchNormInferenceGrad, NHWC, half, float);
INSTANTIATE_GRAD_KERNEL(BatchNormInferenceGrad, NHWC, float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_GRAD_KERNEL(BatchNormInferenceGrad, NCHW, double, double);
INSTANTIATE_GRAD_KERNEL(BatchNormInferenceGrad, NHWC, double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_GRAD_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T, AccT)           \
  template <>                                     \
  void BatchNormExpectation<T, AccT, MPSContext>( \
      const int N,                                \
      const int C,                                \
      const int S,                                \
      const float normalizer,                     \
      const string& data_format,                  \
      const T* x,                                 \
      AccT* ex,                                   \
      AccT* ex2,                                  \
      MPSContext* ctx) {                          \
    NOT_IMPLEMENTED;                              \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                                        \
  template <>                                                                  \
  void BatchNorm<T, AccT, MPSContext>(                                         \
      const int N,                                                             \
      const int C,                                                             \
      const int S,                                                             \
      const string& data_format,                                               \
      const T* x,                                                              \
      const AccT* mu,                                                          \
      const AccT* rsig,                                                        \
      const AccT* gamma,                                                       \
      const AccT* beta,                                                        \
      AccT* scale,                                                             \
      AccT* bias,                                                              \
      T* y,                                                                    \
      MPSContext* ctx) {                                                       \
    auto kernel1 = MPSKernel::TypedString<AccT>("BatchNormFuseParams");        \
    auto kernel2 = MPSKernel::TypedString<T>("BatchNormAffine" + data_format); \
    size_t scale_offset = 0, bias_offset = 0;                                  \
    if (scale == rsig) {                                                       \
      scale_offset = sizeof(AccT) * C, bias_offset = sizeof(AccT) * C * 2;     \
    } else if (scale == bias) {                                                \
      bias_offset = sizeof(AccT) * C;                                          \
    }                                                                          \
    auto* command_buffer = ctx->mps_stream()->command_buffer();                \
    {                                                                          \
      auto* encoder = [command_buffer computeCommandEncoder];                  \
      auto* pso = MPSKernel(kernel1, METAL_SHADERS).GetState(ctx);             \
      [encoder setComputePipelineState:pso];                                   \
      [encoder setBuffer:id<MTLBuffer>(mu) offset:0 atIndex:0];                \
      [encoder setBuffer:id<MTLBuffer>(rsig) offset:0 atIndex:1];              \
      [encoder setBuffer:id<MTLBuffer>(gamma) offset:0 atIndex:2];             \
      [encoder setBuffer:id<MTLBuffer>(beta) offset:0 atIndex:3];              \
      [encoder setBuffer:id<MTLBuffer>(scale) offset:scale_offset atIndex:4];  \
      [encoder setBuffer:id<MTLBuffer>(bias) offset:bias_offset atIndex:5];    \
      MPSDispatchThreads(C, encoder, pso);                                     \
      [encoder endEncoding];                                                   \
      [encoder release];                                                       \
    }                                                                          \
    {                                                                          \
      const uint arg1 = C, arg2 = S;                                           \
      auto args = vector<MPSConstant>({                                        \
          MPSConstant(&arg1, MTLDataTypeUInt, 0),                              \
          MPSConstant(&arg2, MTLDataTypeUInt, 1),                              \
      });                                                                      \
      if (data_format == "NHWC") args.pop_back();                              \
      auto* encoder = [command_buffer computeCommandEncoder];                  \
      auto* pso = MPSKernel(kernel2, METAL_SHADERS).GetState(ctx, args);       \
      [encoder setComputePipelineState:pso];                                   \
      [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                 \
      [encoder setBuffer:id<MTLBuffer>(scale) offset:scale_offset atIndex:1];  \
      [encoder setBuffer:id<MTLBuffer>(bias) offset:bias_offset atIndex:2];    \
      [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:3];                 \
      MPSDispatchThreads((N * C * S), encoder, pso);                           \
      [encoder endEncoding];                                                   \
      [encoder release];                                                       \
    }                                                                          \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                                 \
  template <>                                                                \
  void BatchNormWGrad<T, AccT, MPSContext>(                                  \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const T* x,                                                            \
      const AccT* mu,                                                        \
      const AccT* rsig,                                                      \
      const T* dy,                                                           \
      AccT* dgamma,                                                          \
      AccT* dbeta,                                                           \
      MPSContext* ctx) {                                                     \
    auto kernel = MPSKernel::TypedString<T>("BatchNormWGrad" + data_format); \
    const uint arg1 = C, arg2 = S, arg3 = N * S;                             \
    const auto dbeta_offset = dgamma == dbeta ? sizeof(AccT) * C : 0;        \
    auto args = vector<MPSConstant>({                                        \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                              \
        MPSConstant(&arg3, MTLDataTypeUInt, 2),                              \
        MPSConstant(&arg2, MTLDataTypeUInt, 1),                              \
    });                                                                      \
    if (data_format == "NHWC") args.pop_back();                              \
    auto* command_buffer = ctx->mps_stream()->command_buffer();              \
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
    [encoder setBuffer:id<MTLBuffer>(dbeta) offset:dbeta_offset atIndex:5];  \
    [encoder setThreadgroupMemoryLength:storage_size atIndex:0];             \
    MPSDispatchThreads(C, block_threads, encoder, pso);                      \
    [encoder endEncoding];                                                   \
    [encoder release];                                                       \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                                \
  template <>                                                               \
  void BatchNormTrainingGrad<T, AccT, MPSContext>(                          \
      const int N,                                                          \
      const int C,                                                          \
      const int S,                                                          \
      const float normalizer,                                               \
      const string& data_format,                                            \
      const T* x,                                                           \
      const AccT* mu,                                                       \
      const AccT* rsig,                                                     \
      const AccT* gamma,                                                    \
      const AccT* dgamma,                                                   \
      const AccT* dbeta,                                                    \
      const T* dy,                                                          \
      T* dx,                                                                \
      MPSContext* ctx) {                                                    \
    auto kernel = "BatchNormTrainingGrad" + data_format;                    \
    kernel = MPSKernel::TypedString<T>(kernel);                             \
    const uint arg1 = C, arg2 = S;                                          \
    const auto dbeta_offset = dgamma == dbeta ? sizeof(AccT) * C : 0;       \
    auto args = vector<MPSConstant>({                                       \
        MPSConstant(&normalizer, MTLDataTypeFloat, 3),                      \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                             \
        MPSConstant(&arg2, MTLDataTypeUInt, 1),                             \
    });                                                                     \
    if (data_format == "NHWC") args.pop_back();                             \
    auto* command_buffer = ctx->mps_stream()->command_buffer();             \
    auto* encoder = [command_buffer computeCommandEncoder];                 \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);       \
    [encoder setComputePipelineState:pso];                                  \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];                \
    [encoder setBuffer:id<MTLBuffer>(mu) offset:0 atIndex:1];               \
    [encoder setBuffer:id<MTLBuffer>(rsig) offset:0 atIndex:2];             \
    [encoder setBuffer:id<MTLBuffer>(gamma) offset:0 atIndex:3];            \
    [encoder setBuffer:id<MTLBuffer>(dgamma) offset:0 atIndex:4];           \
    [encoder setBuffer:id<MTLBuffer>(dbeta) offset:dbeta_offset atIndex:5]; \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:6];               \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:7];               \
    MPSDispatchThreads((N * C * S), encoder, pso);                          \
    [encoder endEncoding];                                                  \
    [encoder release];                                                      \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                          \
  template <>                                                         \
  void BatchNormInferenceGrad<T, AccT, MPSContext>(                   \
      const int N,                                                    \
      const int C,                                                    \
      const int S,                                                    \
      const string& data_format,                                      \
      const T* x,                                                     \
      const AccT* mu,                                                 \
      const AccT* rsig,                                               \
      const AccT* gamma,                                              \
      const T* dy,                                                    \
      AccT* dgamma,                                                   \
      AccT* dbeta,                                                    \
      T* dx,                                                          \
      MPSContext* ctx) {                                              \
    if (dgamma != nullptr) {                                          \
      BatchNormWGrad(                                                 \
          N, C, S, data_format, x, mu, rsig, dy, dgamma, dbeta, ctx); \
    }                                                                 \
    auto kernel = "BatchNormInferenceGrad" + data_format;             \
    kernel = MPSKernel::TypedString<T>(kernel);                       \
    const uint arg1 = C, arg2 = S;                                    \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                       \
        MPSConstant(&arg2, MTLDataTypeUInt, 1),                       \
    });                                                               \
    if (data_format == "NHWC") args.pop_back();                       \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(rsig) offset:0 atIndex:0];       \
    [encoder setBuffer:id<MTLBuffer>(gamma) offset:0 atIndex:1];      \
    [encoder setBuffer:id<MTLBuffer>(dy) offset:0 atIndex:2];         \
    [encoder setBuffer:id<MTLBuffer>(dx) offset:0 atIndex:3];         \
    MPSDispatchThreads((N * C * S), encoder, pso);                    \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
