#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _SmoothL1(const int nthreads, const T beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const T val = x[i];
    const T abs_val = abs(val);
    y[i] = abs_val < beta ? T(.5) * val * val / beta : abs_val - T(.5) * beta;
  }
}

__global__ void
_SmoothL1(const int nthreads, const float beta, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float val = __half2float(x[i]);
    const float abs_val = abs(val);
    y[i] = __float2half(
        abs_val < beta ? .5f * val * val / beta : abs_val - .5f * beta);
  }
}

template <typename T>
__global__ void
_SmoothL1Grad(const int nthreads, const T beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const T val = x[i];
    const T abs_val = abs(val);
    y[i] = abs_val < beta ? val / beta : (T)((val > T(0)) - (val < T(0)));
  }
}

__global__ void
_SmoothL1Grad(const int nthreads, const float beta, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const float val = x[i];
    const float abs_val = abs(val);
    y[i] = __float2half(
        abs_val < beta ? val / beta : (float)((val > 0.f) - (val < 0.f)));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void SmoothL1<float16, CUDAContext>(
    const int count,
    const float beta,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _SmoothL1<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      beta,
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y));
}

template <>
void SmoothL1Grad<float16, CUDAContext>(
    const int count,
    const float beta,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _SmoothL1Grad<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count,
      beta,
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y));
} // SmoothL1Grad

#define DEFINE_KERNEL_LAUNCHER(name, T)                                        \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int count, const float beta, const T* x, T* y, CUDAContext* ctx) { \
    _##name<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
        count, (T)beta, x, y);                                                 \
  }

DEFINE_KERNEL_LAUNCHER(SmoothL1, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1, double);

DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
