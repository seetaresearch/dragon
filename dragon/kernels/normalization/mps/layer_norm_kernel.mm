#include "dragon/kernels/normalization/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]];   // C
constant float float_arg1 [[function_constant(1)]]; // eps

template <typename T>
T WarpReduceSum(T val, const ushort warp_size) {
  for (ushort offset = warp_size / 2; offset > 0; offset /= 2) {
    val += simd_shuffle_down(val, offset);
  }
  return val;
}

template <typename T, typename AccT>
kernel void LayerNorm(
    device const T* x,
    device const AccT* gamma,
    device const AccT* beta,
    device AccT* mu,
    device AccT* rsig,
    device T* y,
    threadgroup AccT* storage [[threadgroup(0)]],
    const uint block_size [[threads_per_threadgroup]],
    const uint warp_size [[threads_per_simdgroup]],
    const uint i [[threadgroup_position_in_grid]],
    const uint jstart [[thread_position_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint block_warps = block_size / warp_size;
  threadgroup AccT block_mu, block_rsig;
  threadgroup AccT* m_storage = storage;
  threadgroup AccT* v_storage = storage + block_warps;
  const AccT scale = AccT(1) / AccT(uint_arg1);
  AccT m_val = AccT(0), v_val = AccT(0);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const AccT val = AccT(x[i * uint_arg1 + j]);
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
    mu[i] = block_mu = m_val = m_val * scale;
    rsig[i] = block_rsig = rsqrt(v_val * scale - m_val * m_val + float_arg1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = jstart; j < uint_arg1; j += block_size) {
    const uint index = i * uint_arg1 + j;
    m_val = AccT(x[index]);
    m_val = (m_val - block_mu) * block_rsig;
    y[index] = T(fma(m_val, gamma[j], beta[j]));
  }
}

#define INSTANTIATE_KERNEL(T, AccT) \
  template [[host_name("LayerNorm_"#T)]] \
  kernel void LayerNorm(device const T*, device const AccT*, \
                        device const AccT*, device AccT*, device AccT*, \
                        device T*, threadgroup AccT*, \
                        uint, uint, uint, uint, uint, uint);

INSTANTIATE_KERNEL(half, float);
INSTANTIATE_KERNEL(float, float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(double, double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                               \
  template <>                                                         \
  void LayerNorm<T, AccT, MPSContext>(                                \
      const int N,                                                    \
      const int C,                                                    \
      const float epsilon,                                            \
      const T* x,                                                     \
      const AccT* gamma,                                              \
      const AccT* beta,                                               \
      AccT* mu,                                                       \
      AccT* rsig,                                                     \
      T* y,                                                           \
      MPSContext* ctx) {                                              \
    auto kernel = MPSKernel::TypedString<T>("LayerNorm");             \
    const uint arg1 = C;                                              \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&arg1, MTLDataTypeUInt, 0),                       \
        MPSConstant(&epsilon, MTLDataTypeFloat, 1),                   \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    const int block_threads = MPSGetBlockReduceThreads(C, pso);       \
    const int block_warps = block_threads / pso.threadExecutionWidth; \
    const auto storage_size = sizeof(AccT) * block_warps * 2;         \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];          \
    [encoder setBuffer:id<MTLBuffer>(gamma) offset:0 atIndex:1];      \
    [encoder setBuffer:id<MTLBuffer>(beta) offset:0 atIndex:2];       \
    [encoder setBuffer:id<MTLBuffer>(mu) offset:0 atIndex:3];         \
    [encoder setBuffer:id<MTLBuffer>(rsig) offset:0 atIndex:4];       \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:5];          \
    [encoder setThreadgroupMemoryLength:storage_size atIndex:0];      \
    MPSDispatchThreads(N, block_threads, encoder, pso);               \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);

} // namespace kernels

} // namespace dragon
