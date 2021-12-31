#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename CopyT>
__global__ void _RMSprop(
    const int N,
    const T lr,
    const T momentum,
    const T alpha,
    const T eps,
    const T wd,
    const T* x,
    const T* g,
    T* m,
    T* v,
    T* y,
    CopyT* y_copy) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T gi = wd > T(0) ? fma(wd, x[i], g[i]) : g[i];
    const T vi = v[i] = fma(alpha, v[i], (T(1) - alpha) * gi * gi);
    const T mi = m[i] = fma(momentum, m[i], gi / (std::sqrt(vi) + eps));
    y[i] -= lr * mi;
    if (y_copy != nullptr) {
      y_copy[i] = convert::To<CopyT>(y[i]);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT)                        \
  template <>                                                         \
  void name<T, CopyT, CUDAContext>(                                   \
      const int N,                                                    \
      const float lr,                                                 \
      const float momentum,                                           \
      const float alpha,                                              \
      const float eps,                                                \
      const float wd,                                                 \
      const T* x,                                                     \
      const T* g,                                                     \
      T* m,                                                           \
      T* v,                                                           \
      T* y,                                                           \
      CopyT* y_copy,                                                  \
      CUDAContext* ctx) {                                             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                            \
        convert::To<T>(lr),                                           \
        convert::To<T>(momentum),                                     \
        convert::To<T>(alpha),                                        \
        convert::To<T>(eps),                                          \
        convert::To<T>(wd),                                           \
        x,                                                            \
        g,                                                            \
        m,                                                            \
        v,                                                            \
        y,                                                            \
        reinterpret_cast<math::ScalarType<CopyT>::type*>(y_copy));    \
  }

DEFINE_KERNEL_LAUNCHER(RMSprop, float, float16);
DEFINE_KERNEL_LAUNCHER(RMSprop, float, float);
DEFINE_KERNEL_LAUNCHER(RMSprop, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
