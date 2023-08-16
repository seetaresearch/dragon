#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_SetOneHot(const int N, const int depth, const T value, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i * depth + int(convert::To<AccT>(x[i]))] = value;
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void SetOneHot<T, CUDAContext>(                                     \
      const int N,                                                    \
      const int depth,                                                \
      const float value,                                              \
      const T* x,                                                     \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    _SetOneHot<                                                       \
        math::Traits<T>::scalar_type,                                 \
        math::Traits<T>::accumulator_type>                            \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
            N,                                                        \
            depth,                                                    \
            convert::To<math::Traits<T>::scalar_type>(value),         \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x), \
            reinterpret_cast<math::Traits<T>::scalar_type*>(y));      \
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
