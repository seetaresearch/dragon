#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Adam(
    const int N,
    const T lr,
    const T beta1,
    const T beta2,
    const T eps,
    T* g,
    T* m,
    T* v) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    T gi = g[i];
    T mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
    T vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    g[i] = lr * mi / (sqrt(vi) + eps);
  }
}

template <typename T>
__global__ void _AdamW(
    const int N,
    const T lr,
    const T beta1,
    const T beta2,
    const T eps,
    const T wd,
    const T* x,
    T* g,
    T* m,
    T* v) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    T gi = g[i];
    T mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
    T vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    g[i] = lr * mi / (sqrt(vi) + eps) + wd * x[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Adam<float, CUDAContext>(
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    float* g,
    float* m,
    float* v,
    CUDAContext* ctx) {
  _Adam<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, lr, beta1, beta2, eps, g, m, v);
}

template <>
void AdamW<float, CUDAContext>(
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float* x,
    float* g,
    float* m,
    float* v,
    CUDAContext* ctx) {
  _AdamW<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, lr, beta1, beta2, eps, wd, x, g, m, v);
}

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
