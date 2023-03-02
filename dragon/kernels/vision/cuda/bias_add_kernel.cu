#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_BiasAdd(const int NxC, const int C, const T* x, const T* bias, T* y) {
  const math::PlusFunctor<T> functor;
  CUDA_1D_KERNEL_LOOP(i, NxC) {
    y[i] = functor(x[i], __ldg(bias + i % C));
  }
}

template <typename T>
__global__ void _BiasAdd(
    const int NxCxS,
    const int S,
    const int C,
    const T* x,
    const T* bias,
    T* y) {
  const math::PlusFunctor<T> functor;
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    y[i] = functor(x[i], __ldg(bias + i / S % C));
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void BiasAdd<T, CUDAContext>(                                              \
      const int N,                                                           \
      const int S,                                                           \
      const int C,                                                           \
      const T* x,                                                            \
      const T* bias,                                                         \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    const auto NxCxS = N * C * S;                                            \
    if (S == 1) {                                                            \
      _BiasAdd<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          NxCxS,                                                             \
          C,                                                                 \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),             \
          reinterpret_cast<const math::ScalarType<T>::type*>(bias),          \
          reinterpret_cast<math::ScalarType<T>::type*>(y));                  \
    } else {                                                                 \
      _BiasAdd<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          NxCxS,                                                             \
          S,                                                                 \
          C,                                                                 \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),             \
          reinterpret_cast<const math::ScalarType<T>::type*>(bias),          \
          reinterpret_cast<math::ScalarType<T>::type*>(y));                  \
    }                                                                        \
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