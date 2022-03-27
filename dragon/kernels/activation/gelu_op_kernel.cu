#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _Gelu(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    y[i] = convert::To<T>(val * normcdf(val));
  }
}

template <typename T, typename AccT>
__global__ void _GeluGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    dx[i] = convert::To<T>(
        convert::To<AccT>(dy[i]) *
        fma(AccT(0.3989422804014327) * val,
            exp(val * val * AccT(-0.5)),
            normcdf(val)));
  }
}

template <typename T, typename AccT>
__global__ void _ApproxGelu(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    y[i] = fma(val,
               tanh(
                   AccT(0.7978845608028654) *
                   fma(AccT(0.044715), val * val * val, val)),
               val) *
        AccT(0.5);
  }
}

template <typename T, typename AccT>
__global__ void _ApproxGeluGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    const AccT val2 = tanh(
        AccT(0.7978845608028654) * fma(AccT(0.044715), val * val * val, val));
    dx[i] = convert::To<T>(
        convert::To<AccT>(dy[i]) * AccT(0.5) *
        fma(fma(-val, val2 * val2, val),
            fma(AccT(0.10703222440890037), val * val, AccT(0.7978845608028654)),
            val2 + AccT(1)));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

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

DEFINE_KERNEL_LAUNCHER(Gelu, float16);
DEFINE_KERNEL_LAUNCHER(Gelu, float);
DEFINE_KERNEL_LAUNCHER(Gelu, double);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float16);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, double);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
