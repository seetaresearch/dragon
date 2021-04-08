#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_SetOneHot(const int N, const int depth, const T value, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i * depth + int(x[i])] = value;
  }
}

template <>
__global__ void _SetOneHot<half>(
    const int N,
    const int depth,
    const half value,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i * depth + int(__half2float(x[i]))] = value;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void SetOneHot<T, CUDAContext>(                                        \
      const int N,                                                       \
      const int depth,                                                   \
      const float value,                                                 \
      const T* x,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    _SetOneHot<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                               \
        depth,                                                           \
        convert::To<math::ScalarType<T>::type>(value),                   \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),           \
        reinterpret_cast<math::ScalarType<T>::type*>(y));                \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
