#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _Gelu(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    y[i] = v * normcdf(v);
  }
}

template <typename T, typename AccT>
__global__ void _ApproxGelu(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    const AccT s = tanh(AccT(0.797884) * fma(AccT(0.044715), v * v * v, v));
    y[i] = fma(v, s, v) * AccT(0.5);
  }
}

template <typename T, typename AccT>
__global__ void _GeluGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    dx[i] = convert::To<AccT>(dy[i]) *
        fma(AccT(0.398942) * v, exp(v * v * AccT(-0.5)), normcdf(v));
  }
}

template <typename T, typename AccT>
__global__ void _ApproxGeluGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT v = convert::To<AccT>(x[i]);
    const AccT s = tanh(AccT(0.797884) * fma(AccT(0.044715), v * v * v, v));
    dx[i] = convert::To<AccT>(dy[i]) * AccT(0.5) *
        fma(fma(-v, s * s, v),
            fma(AccT(0.107032), v * v, AccT(0.797884)),
            s + AccT(1));
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                        \
  template <>                                                                  \
  void name<T, CUDAContext>(const int N, const T* x, T* y, CUDAContext* ctx) { \
    _##name<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type>   \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(             \
            N,                                                                 \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),          \
            reinterpret_cast<math::Traits<T>::scalar_type*>(y));               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                 \
  template <>                                                                \
  void name<T, CUDAContext>(                                                 \
      const int N, const T* dy, const T* x, T* dx, CUDAContext* ctx) {       \
    _##name<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(           \
            N,                                                               \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),       \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),        \
            reinterpret_cast<math::Traits<T>::scalar_type*>(dx));            \
  }

DEFINE_KERNEL_LAUNCHER(Gelu, float16);
DEFINE_KERNEL_LAUNCHER(Gelu, bfloat16);
DEFINE_KERNEL_LAUNCHER(Gelu, float);
DEFINE_KERNEL_LAUNCHER(Gelu, double);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float16);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, bfloat16);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, double);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
