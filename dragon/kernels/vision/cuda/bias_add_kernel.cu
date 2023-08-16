#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_BiasAdd(const int NxC, const int C, const T* x, const T* bias, T* y) {
  const auto add = math::PlusFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, NxC) {
    y[i] = add(x[i], math::utils::LDG(bias + i % C));
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
  const auto add = math::PlusFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    y[i] = add(x[i], math::utils::LDG(bias + i / S % C));
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void BiasAdd<T, CUDAContext>(                                               \
      const int N,                                                            \
      const int S,                                                            \
      const int C,                                                            \
      const T* x,                                                             \
      const T* bias,                                                          \
      T* y,                                                                   \
      CUDAContext* ctx) {                                                     \
    using ScalarT = math::Traits<T>::scalar_type;                             \
    const auto NxCxS = N * C * S;                                             \
    if (S == 1) {                                                             \
      _BiasAdd<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
          NxCxS, C, (const ScalarT*)x, (const ScalarT*)bias, (ScalarT*)y);    \
    } else {                                                                  \
      _BiasAdd<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
          NxCxS, S, C, (const ScalarT*)x, (const ScalarT*)bias, (ScalarT*)y); \
    }                                                                         \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
