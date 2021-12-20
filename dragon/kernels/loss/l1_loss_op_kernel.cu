#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _SmoothL1(const int N, const AccT beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    const AccT abs_val = abs(val);
    y[i] = convert::To<T>(
        abs_val < beta ? AccT(.5) * val * val / beta
                       : abs_val - AccT(.5) * beta);
  }
}

template <typename T, typename AccT>
__global__ void _SmoothL1Grad(const int N, const AccT beta, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(x[i]);
    const AccT abs_val = abs(val);
    y[i] = convert::To<T>(
        abs_val < beta ? val / beta
                       : (AccT)((val > AccT(0)) - (val < AccT(0))));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                                    \
  template <>                                                              \
  void name<T, CUDAContext>(                                               \
      const int N, const float beta, const T* x, T* y, CUDAContext* ctx) { \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
        N,                                                                 \
        convert::To<math::AccmulatorType<T>::type>(beta),                  \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),             \
        reinterpret_cast<math::ScalarType<T>::type*>(y));                  \
  }

DEFINE_KERNEL_LAUNCHER(SmoothL1, float16);
DEFINE_KERNEL_LAUNCHER(SmoothL1, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1, double);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, float16);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
