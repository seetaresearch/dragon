#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_NesterovUpdate(const int nthreads, const T lr, const T momentum, T* g, T* m);

template <>
__global__ void _NesterovUpdate<float>(
    const int nthreads,
    const float lr,
    const float momentum,
    float* g,
    float* m) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    float mi = m[i];
    float mi_new = m[i] = momentum * mi + lr * g[i];
    g[i] = fmaf(momentum, mi_new - mi, mi_new);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void NesterovUpdate<float, CUDAContext>(
    const int count,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CUDAContext* ctx) {
  _NesterovUpdate<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, lr, momentum, g, m);
}

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
