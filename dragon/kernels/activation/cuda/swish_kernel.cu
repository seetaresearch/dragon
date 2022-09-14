#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _Silu(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    y[i] = convert::To<T>(val / (AccT(1) + exp(-val)));
  }
}

template <typename T, typename AccT>
__global__ void _HardSwish(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    const AccT s_val = fma(val, AccT(0.1666666666666667), AccT(0.5));
    y[i] = convert::To<T>(val * max(AccT(0), min(AccT(1), s_val)));
  }
}

template <typename T, typename AccT>
__global__ void _SiluGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    const AccT s_val = AccT(1) / (AccT(1) + exp(-val));
    dx[i] = convert::To<T>(
        convert::To<AccT>(dy[i]) * s_val * (val + AccT(1) - val * s_val));
  }
}

template <typename T, typename AccT>
__global__ void _HardSwishGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    dx[i] = (val < AccT(-3)) ? convert::To<T>(AccT(0))
        : (val < AccT(3))    ? convert::To<T>(
                                convert::To<AccT>(dy[i]) *
                                fma(val, AccT(0.3333333333333333), AccT(0.5)))
                          : dy[i];
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                        \
  template <>                                                                  \
  void name<T, CUDAContext>(const int N, const T* x, T* y, CUDAContext* ctx) { \
    _##name<math::ScalarType<T>::type, math::AccumulatorType<T>::type>         \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(             \
            N,                                                                 \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),             \
            reinterpret_cast<math::ScalarType<T>::type*>(y));                  \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                           \
  template <>                                                          \
  void name<T, CUDAContext>(                                           \
      const int N, const T* dy, const T* x, T* dx, CUDAContext* ctx) { \
    _##name<math::ScalarType<T>::type, math::AccumulatorType<T>::type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
            N,                                                         \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy),    \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),     \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));         \
  }

DEFINE_KERNEL_LAUNCHER(Silu, float16);
DEFINE_KERNEL_LAUNCHER(Silu, float);
DEFINE_KERNEL_LAUNCHER(Silu, double);
DEFINE_KERNEL_LAUNCHER(HardSwish, float16);
DEFINE_KERNEL_LAUNCHER(HardSwish, float);
DEFINE_KERNEL_LAUNCHER(HardSwish, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
