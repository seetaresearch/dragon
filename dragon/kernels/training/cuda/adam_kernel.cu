#include "dragon/kernels/training/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename CopyT>
__global__ void _Adam(
    const int N,
    const T lr,
    const T beta1,
    const T beta2,
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
    const T mi = m[i] = fma(beta1, m[i], (T(1) - beta1) * gi);
    const T vi = v[i] = fma(beta2, v[i], (T(1) - beta2) * gi * gi);
    y[i] -= lr * mi / (sqrt(vi) + eps);
    if (y_copy != nullptr) y_copy[i] = convert::To<CopyT>(y[i]);
  }
}

template <typename T, typename CopyT>
__global__ void _AdamW(
    const int N,
    const T lr,
    const T beta1,
    const T beta2,
    const T eps,
    const T wd,
    const T* x,
    const T* g,
    T* m,
    T* v,
    T* y,
    CopyT* y_copy) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T gi = g[i];
    const T mi = m[i] = fma(beta1, m[i], (T(1) - beta1) * gi);
    const T vi = v[i] = fma(beta2, v[i], (T(1) - beta2) * gi * gi);
    y[i] -= wd > T(0) ? fma(wd, x[i], lr * mi / (sqrt(vi) + eps))
                      : lr * mi / (sqrt(vi) + eps);
    if (y_copy != nullptr) y_copy[i] = convert::To<CopyT>(y[i]);
  }
}

template <typename T, typename CopyT>
__global__ void _AdamW(
    const int N,
    const T lr,
    const T beta1,
    const T beta2,
    const T eps,
    const T wd,
    const T* lrs,
    const T* x,
    const T* g,
    T* m,
    T* v,
    T* y,
    CopyT* y_copy) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T gi = g[i]; // clang-format off
    const T mi = m[i] = fma(beta1, m[i], (T(1) - beta1) * gi);
    const T vi = v[i] = fma(beta2, v[i], (T(1) - beta2) * gi * gi);
    y[i] -= wd > T(0) ? fma(wd * __ldg(lrs + i), x[i], lr * __ldg(lrs + i) * mi / (sqrt(vi) + eps))
                      : lr * lrs[i] * mi / (sqrt(vi) + eps); // clang-format on
    if (y_copy != nullptr) y_copy[i] = convert::To<CopyT>(y[i]);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT)                        \
  template <>                                                         \
  void name<T, CopyT, CUDAContext>(                                   \
      const int N,                                                    \
      const float lr,                                                 \
      const float beta1,                                              \
      const float beta2,                                              \
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
        convert::To<T>(beta1),                                        \
        convert::To<T>(beta2),                                        \
        convert::To<T>(eps),                                          \
        convert::To<T>(wd),                                           \
        x,                                                            \
        g,                                                            \
        m,                                                            \
        v,                                                            \
        y,                                                            \
        reinterpret_cast<math::Traits<CopyT>::scalar_type*>(y_copy)); \
  }

DEFINE_KERNEL_LAUNCHER(Adam, float, float16);
DEFINE_KERNEL_LAUNCHER(Adam, float, bfloat16);
DEFINE_KERNEL_LAUNCHER(Adam, float, float);
DEFINE_KERNEL_LAUNCHER(Adam, double, double);
DEFINE_KERNEL_LAUNCHER(AdamW, float, float16);
DEFINE_KERNEL_LAUNCHER(AdamW, float, bfloat16);
DEFINE_KERNEL_LAUNCHER(AdamW, float, float);
DEFINE_KERNEL_LAUNCHER(AdamW, double, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT)                        \
  template <>                                                         \
  void name<T, CopyT, CUDAContext>(                                   \
      const int N,                                                    \
      const float lr,                                                 \
      const float beta1,                                              \
      const float beta2,                                              \
      const float eps,                                                \
      const float wd,                                                 \
      const T* lrs,                                                   \
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
        convert::To<T>(beta1),                                        \
        convert::To<T>(beta2),                                        \
        convert::To<T>(eps),                                          \
        convert::To<T>(wd),                                           \
        lrs,                                                          \
        x,                                                            \
        g,                                                            \
        m,                                                            \
        v,                                                            \
        y,                                                            \
        reinterpret_cast<math::Traits<CopyT>::scalar_type*>(y_copy)); \
  }

DEFINE_KERNEL_LAUNCHER(AdamW, float, float16);
DEFINE_KERNEL_LAUNCHER(AdamW, float, bfloat16);
DEFINE_KERNEL_LAUNCHER(AdamW, float, float);
DEFINE_KERNEL_LAUNCHER(AdamW, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
