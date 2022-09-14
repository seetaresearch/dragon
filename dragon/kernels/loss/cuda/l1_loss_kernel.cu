#include "dragon/kernels/loss/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_SmoothL1(const int N, const AccT beta, const T* diff, T* loss) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(diff[i]);
    const AccT abs_val = abs(val);
    loss[i] = convert::To<T>(
        abs_val < beta ? AccT(.5) * val * val / beta
                       : abs_val - AccT(.5) * beta);
  }
}

template <typename T, typename AccT>
__global__ void
_SmoothL1Grad(const int N, const AccT beta, const T* diff, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(diff[i]);
    const AccT abs_val = abs(val);
    dx[i] = convert::To<T>(
        abs_val < beta ? val / beta
                       : (AccT)((val > AccT(0)) - (val < AccT(0))));
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const float beta,                                               \
      const T* diff,                                                  \
      T* loss,                                                        \
      CUDAContext* ctx) {                                             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                            \
        convert::To<math::AccumulatorType<T>::type>(beta),            \
        reinterpret_cast<const math::ScalarType<T>::type*>(diff),     \
        reinterpret_cast<math::ScalarType<T>::type*>(loss));          \
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
