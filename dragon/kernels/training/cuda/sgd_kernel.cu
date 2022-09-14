#include "dragon/kernels/training/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename CopyT>
__global__ void _MomentumSGD(
    const int N,
    const T lr,
    const T momentum,
    const T wd,
    const T* x,
    const T* g,
    T* m,
    T* y,
    CopyT* y_copy) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T gi = wd > T(0) ? fma(wd, x[i], g[i]) : g[i];
    const T mi = m[i] = fma(momentum, m[i], gi);
    y[i] -= lr * mi;
    if (y_copy != nullptr) {
      y_copy[i] = convert::To<CopyT>(y[i]);
    }
  }
}

template <typename T, typename CopyT>
__global__ void _NesterovSGD(
    const int N,
    const T lr,
    const T momentum,
    const T wd,
    const T* x,
    const T* g,
    T* m,
    T* y,
    CopyT* y_copy) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T gi = wd > T(0) ? fma(wd, x[i], g[i]) : g[i];
    const T mi = m[i] = fma(momentum, m[i], gi);
    y[i] -= lr * fma(momentum, mi, gi);
    if (y_copy != nullptr) {
      y_copy[i] = convert::To<CopyT>(y[i]);
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT)                        \
  template <>                                                         \
  void name<T, CopyT, CUDAContext>(                                   \
      const int N,                                                    \
      const float lr,                                                 \
      const float momentum,                                           \
      const float wd,                                                 \
      const T* x,                                                     \
      const T* g,                                                     \
      T* m,                                                           \
      T* y,                                                           \
      CopyT* y_copy,                                                  \
      CUDAContext* ctx) {                                             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                            \
        convert::To<T>(lr),                                           \
        convert::To<T>(momentum),                                     \
        convert::To<T>(wd),                                           \
        x,                                                            \
        g,                                                            \
        m,                                                            \
        y,                                                            \
        reinterpret_cast<math::ScalarType<CopyT>::type*>(y_copy));    \
  }

DEFINE_KERNEL_LAUNCHER(MomentumSGD, float, float16);
DEFINE_KERNEL_LAUNCHER(MomentumSGD, float, float);
DEFINE_KERNEL_LAUNCHER(MomentumSGD, double, double);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, float, float16);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, float, float);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
