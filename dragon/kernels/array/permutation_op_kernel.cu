#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

__global__ void _Sequence(const int nthreads, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __float2half(float(i));
  }
}

} // namespace

template <>
void Permutation<float16, CUDAContext>(
    const int count,
    float16* y,
    uint32_t* r,
    CUDAContext* ctx) {
  math::Random(count, r, ctx);
  auto values = thrust::device_ptr<half>(reinterpret_cast<half*>(y));
  auto keys = thrust::device_ptr<uint32_t>(r);
  auto policy = thrust::cuda::par.on(ctx->cuda_stream());
  _Sequence<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, reinterpret_cast<half*>(y));
  thrust::sort_by_key(policy, keys, keys + count, values);
}

#define DEFINE_KERNEL_LAUNCHER(T)                             \
  template <>                                                 \
  void Permutation<T, CUDAContext>(                           \
      const int count, T* y, uint32_t* r, CUDAContext* ctx) { \
    math::Random(count, r, ctx);                              \
    auto values = thrust::device_ptr<T>(y);                   \
    auto keys = thrust::device_ptr<uint32_t>(r);              \
    auto policy = thrust::cuda::par.on(ctx->cuda_stream());   \
    thrust::sequence(policy, values, values + count);         \
    thrust::sort_by_key(policy, keys, keys + count, values);  \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
