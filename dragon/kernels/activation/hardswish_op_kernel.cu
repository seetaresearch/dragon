#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _HardSwish(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    const AccT s_val = fma(val, AccT(0.1666666666666667), AccT(0.5));
    y[i] = convert::To<T>(val * max(AccT(0), min(AccT(1), s_val)));
  }
}

template <typename T, typename AccT>
__global__ void _HardSwishGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    dx[i] = (val < AccT(-3))
        ? convert::To<T>(AccT(0))
        : (val < AccT(3)) ? convert::To<T>(
                                convert::To<AccT>(dy[i]) *
                                fma(val, AccT(0.3333333333333333), AccT(0.5)))
                          : dy[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void HardSwish<T, CUDAContext>(                                        \
      const int N, const T* x, T* y, CUDAContext* ctx) {                 \
    _HardSwish<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(       \
            N,                                                           \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),       \
            reinterpret_cast<math::ScalarType<T>::type*>(y));            \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void HardSwishGrad<T, CUDAContext>(                                        \
      const int N, const T* dy, const T* x, T* dx, CUDAContext* ctx) {       \
    _HardSwishGrad<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(           \
            N,                                                               \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy),          \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),           \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));               \
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
