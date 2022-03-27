#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_BiasAdd(const int NxC, const int C, const T* x, const T* bias, T* y) {
  CUDA_1D_KERNEL_LOOP(i, NxC) {
    y[i] = convert::To<T>(
        convert::To<AccT>(x[i]) + convert::To<AccT>(__ldg(bias + i % C)));
  }
}

template <typename T, typename AccT>
__global__ void _BiasAdd(
    const int NxCxS,
    const int S,
    const int C,
    const T* x,
    const T* bias,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    y[i] = convert::To<T>(
        convert::To<AccT>(x[i]) + convert::To<AccT>(__ldg(bias + (i / S) % C)));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void BiasAdd<T, CUDAContext>(                                           \
      const int N,                                                        \
      const int S,                                                        \
      const int C,                                                        \
      const T* x,                                                         \
      const T* bias,                                                      \
      T* y,                                                               \
      CUDAContext* ctx) {                                                 \
    const auto NxCxS = N * C * S;                                         \
    if (S == 1) {                                                         \
      _BiasAdd<math::ScalarType<T>::type, math::AccumulatorType<T>::type> \
          <<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              NxCxS,                                                      \
              C,                                                          \
              reinterpret_cast<const math::ScalarType<T>::type*>(x),      \
              reinterpret_cast<const math::ScalarType<T>::type*>(bias),   \
              reinterpret_cast<math::ScalarType<T>::type*>(y));           \
    } else {                                                              \
      _BiasAdd<math::ScalarType<T>::type, math::AccumulatorType<T>::type> \
          <<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
              NxCxS,                                                      \
              S,                                                          \
              C,                                                          \
              reinterpret_cast<const math::ScalarType<T>::type*>(x),      \
              reinterpret_cast<const math::ScalarType<T>::type*>(bias),   \
              reinterpret_cast<math::ScalarType<T>::type*>(y));           \
    }                                                                     \
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
