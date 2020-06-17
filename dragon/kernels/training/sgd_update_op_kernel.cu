#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void
_SGDUpdate(const int nthreads, const T lr, const T momentum, T* g, T* m) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    T mi = m[i];
    g[i] = m[i] = momentum * mi + lr * g[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void SGDUpdate<float, CUDAContext>(
    const int count,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CUDAContext* ctx) {
  _SGDUpdate<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, lr, momentum, g, m);
}

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
