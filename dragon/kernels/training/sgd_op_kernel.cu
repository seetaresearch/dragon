#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_MomentumSGD(const int N, const T lr, const T momentum, T* g, T* m) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    T mi = m[i] = fma(momentum, m[i], g[i]);
    g[i] = lr * mi;
  }
}

template <typename T>
__global__ void
_NesterovSGD(const int N, const T lr, const T momentum, T* g, T* m) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    T gi = g[i];
    T mi = m[i] = fma(momentum, m[i], gi);
    g[i] = lr * fma(momentum, mi, gi);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void MomentumSGD<float, CUDAContext>(
    const int N,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CUDAContext* ctx) {
  _MomentumSGD<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, lr, momentum, g, m);
}

template <>
void NesterovSGD<float, CUDAContext>(
    const int N,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CUDAContext* ctx) {
  _NesterovSGD<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, lr, momentum, g, m);
}

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
