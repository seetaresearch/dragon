#include "dragon/kernels/training/op_kernels.h"
#include "dragon/utils/conversions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename CopyT>
void _Adam(
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
  for (int i = 0; i < N; ++i) {
    const T gi = wd > T(0) ? std::fma(wd, x[i], g[i]) : g[i];
    const T mi = m[i] = std::fma(beta1, m[i], (T(1) - beta1) * gi);
    const T vi = v[i] = std::fma(beta2, v[i], (T(1) - beta2) * gi * gi);
    y[i] -= lr * mi / (std::sqrt(vi) + eps);
    if (y_copy != nullptr) y_copy[i] = convert::To<CopyT>(y[i]);
  }
}

template <typename T, typename CopyT>
void _AdamW(
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
  for (int i = 0; i < N; ++i) {
    const T gi = g[i];
    const T mi = m[i] = std::fma(beta1, m[i], (T(1) - beta1) * gi);
    const T vi = v[i] = std::fma(beta2, v[i], (T(1) - beta2) * gi * gi);
    y[i] -= wd > T(0) ? std::fma(wd, x[i], lr * mi / (std::sqrt(vi) + eps))
                      : lr * mi / (std::sqrt(vi) + eps);
    if (y_copy != nullptr) y_copy[i] = convert::To<CopyT>(y[i]);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT) \
  template <>                                  \
  void name<T, CopyT, CPUContext>(             \
      const int N,                             \
      const float lr,                          \
      const float beta1,                       \
      const float beta2,                       \
      const float eps,                         \
      const float wd,                          \
      const T* x,                              \
      const T* g,                              \
      T* m,                                    \
      T* v,                                    \
      T* y,                                    \
      CopyT* y_copy,                           \
      CPUContext* ctx) {                       \
    _##name(                                   \
        N,                                     \
        convert::To<T>(lr),                    \
        convert::To<T>(beta1),                 \
        convert::To<T>(beta2),                 \
        convert::To<T>(eps),                   \
        convert::To<T>(wd),                    \
        x,                                     \
        g,                                     \
        m,                                     \
        v,                                     \
        y,                                     \
        y_copy);                               \
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

} // namespace kernels

} // namespace dragon
