#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

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
__global__ void _SiluGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    const AccT s_val = AccT(1) / (AccT(1) + exp(-val));
    dx[i] = convert::To<T>(
        convert::To<AccT>(dy[i]) * s_val * (val + AccT(1) - val * s_val));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void Silu<T, CUDAContext>(const int N, const T* x, T* y, CUDAContext* ctx) { \
    _Silu<math::ScalarType<T>::type, math::AccmulatorType<T>::type>            \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(             \
            N,                                                                 \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),             \
            reinterpret_cast<math::ScalarType<T>::type*>(y));                  \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void SiluGrad<T, CUDAContext>(                                        \
      const int N, const T* dy, const T* x, T* dx, CUDAContext* ctx) {  \
    _SiluGrad<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
            N,                                                          \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy),     \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),      \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));          \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
