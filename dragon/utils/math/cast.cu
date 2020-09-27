#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/elementwise.h"

namespace dragon {

namespace math {

namespace {

template <typename Tx, typename Ty>
__global__ void _Cast(const int nthreads, const Tx* x, Ty* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = (Ty)x[i];
  }
}

template <>
__global__ void
_Cast<half, float>(const int nthreads, const half* x, float* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __half2float(x[i]);
  }
}

template <>
__global__ void
_Cast<float, half>(const int nthreads, const float* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __float2half(x[i]);
  }
}

template <>
__global__ void _Cast<half, half>(const int nthreads, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = x[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Cast<float16, float, CUDAContext>(
    const int n,
    const float16* x,
    float* y,
    CUDAContext* ctx) {
  _Cast<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), y);
}

template <>
void Cast<float, float16, CUDAContext>(
    const int n,
    const float* x,
    float16* y,
    CUDAContext* ctx) {
  _Cast<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      n, x, reinterpret_cast<half*>(y));
}

template <>
void Cast<float16, float16, CUDAContext>(
    const int n,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _Cast<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
}

#define DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, Ty)                               \
  template <>                                                                \
  void Cast<Tx, Ty, CUDAContext>(                                            \
      const int n, const Tx* x, Ty* y, CUDAContext* ctx) {                   \
    _Cast<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(n, x, y); \
  }

#define DEFINE_KERNEL_LAUNCHER(Tx)             \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, bool);    \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, int8_t);  \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, uint8_t); \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, int);     \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, int64_t); \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, float);   \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, double);

#define DEFINE_FP16_KERNEL_LAUNCHER(T)                                         \
  template <>                                                                  \
  void Cast<float16, T, CUDAContext>(                                          \
      const int n, const float16* x, T* y, CUDAContext* ctx) {                 \
    LOG(FATAL) << "Not Implemented: float16 -> "                               \
               << types::to_string(TypeMeta::Make<T>());                       \
  }                                                                            \
  template <>                                                                  \
  void Cast<T, float16, CUDAContext>(                                          \
      const int n, const T* x, float16* y, CUDAContext* ctx) {                 \
    LOG(FATAL) << "Not Implemented: " << types::to_string(TypeMeta::Make<T>()) \
               << " -> float16";                                               \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_FP16_KERNEL_LAUNCHER(bool);
DEFINE_FP16_KERNEL_LAUNCHER(int8_t);
DEFINE_FP16_KERNEL_LAUNCHER(uint8_t);
DEFINE_FP16_KERNEL_LAUNCHER(int);
DEFINE_FP16_KERNEL_LAUNCHER(int64_t);
DEFINE_FP16_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GENERIC_KERNEL_LAUNCHER
#undef DEFINE_FP16_KERNEL_LAUNCHER

} // namespace math

} // namespace dragon

#endif // USE_CUDA
