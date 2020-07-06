#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

__global__ void _MixedPrecL2Penalty(
    const int nthreads,
    const float alpha,
    const half* x,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[i] += __half2float(x[i]) * alpha;
  }
}

__global__ void _MixedPrecUpdate(const int nthreads, const float* dx, half* x) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    x[i] = __float2half(__half2float(x[i]) - dx[i]);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void MixedPrecL2Penalty<float16, CUDAContext>(
    const int count,
    const float alpha,
    const float16* x,
    float* dx,
    CUDAContext* ctx) {
  _MixedPrecL2Penalty<<<
      CUDA_BLOCKS(count),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(count, alpha, reinterpret_cast<const half*>(x), dx);
}

template <>
void MixedPrecUpdate<float16, CUDAContext>(
    const int count,
    const float* dx,
    float16* x,
    CUDAContext* ctx) {
  _MixedPrecUpdate<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, dx, reinterpret_cast<half*>(x));
}

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
