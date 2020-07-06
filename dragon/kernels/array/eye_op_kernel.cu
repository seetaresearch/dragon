#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/cast.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _SetEye(const int n, const int m, const int k, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i * m + k + i] = T(1);
  }
}

template <>
__global__ void _SetEye<half>(const int n, const int m, const int k, half* y) {
  const half kOne = __float2half(1.f);
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i * m + k + i] = kOne;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Eye<float16, CUDAContext>(
    const int n,
    const int m,
    const int k,
    float16* y,
    CUDAContext* ctx) {
  math::Set(n * m, cast::to<float16>(0.f), y, ctx);
  if (k > 0) {
    if (m - k > 0) {
      _SetEye<<<CUDA_BLOCKS(m - k), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
          m - k, m, k, reinterpret_cast<half*>(y));
    }
  } else {
    if (n + k > 0) {
      _SetEye<<<CUDA_BLOCKS(n + k), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
          n + k, m, 0, reinterpret_cast<half*>(y - k * m));
    }
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void Eye<T, CUDAContext>(                                                   \
      const int n, const int m, const int k, T* y, CUDAContext* ctx) {        \
    math::Set(n* m, T(0), y, ctx);                                            \
    if (k > 0) {                                                              \
      if (m - k > 0) {                                                        \
        _SetEye<<<CUDA_BLOCKS(m - k), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            m - k, m, k, y);                                                  \
      }                                                                       \
    } else {                                                                  \
      if (n + k > 0) {                                                        \
        _SetEye<<<CUDA_BLOCKS(n + k), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            n + k, m, 0, y - k * m);                                          \
      }                                                                       \
    }                                                                         \
  }

DEFINE_KERNEL_LAUNCHER(bool);
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
