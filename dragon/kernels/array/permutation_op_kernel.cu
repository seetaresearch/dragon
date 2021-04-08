#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_thrust.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

__global__ void _Sequence(const int N, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(float(i));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Permutation<float16, CUDAContext>(
    const int N,
    float16* y,
    uint32_t* r,
    CUDAContext* ctx) {
  math::Random(N, r, ctx);
  auto values = thrust::device_ptr<half>(reinterpret_cast<half*>(y));
  auto keys = thrust::device_ptr<uint32_t>(r);
  auto policy = thrust::cuda::par.on(ctx->cuda_stream());
  _Sequence<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, reinterpret_cast<half*>(y));
  thrust::sort_by_key(policy, keys, keys + N, values);
}

#define DEFINE_KERNEL_LAUNCHER(T)                           \
  template <>                                               \
  void Permutation<T, CUDAContext>(                         \
      const int N, T* y, uint32_t* r, CUDAContext* ctx) {   \
    math::Random(N, r, ctx);                                \
    auto values = thrust::device_ptr<T>(y);                 \
    auto keys = thrust::device_ptr<uint32_t>(r);            \
    auto policy = thrust::cuda::par.on(ctx->cuda_stream()); \
    thrust::sequence(policy, values, values + N);           \
    thrust::sort_by_key(policy, keys, keys + N, values);    \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
