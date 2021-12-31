#include "dragon/utils/conversions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename CopyT>
void _RMSprop(
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
  for (int i = 0; i < N; ++i) {
    const T gi = wd > T(0) ? std::fma(wd, x[i], g[i]) : g[i];
    const T vi = v[i] = std::fma(alpha, v[i], (T(1) - alpha) * gi * gi);
    const T mi = m[i] = std::fma(momentum, m[i], gi / (std::sqrt(vi) + eps));
    y[i] -= lr * mi;
    if (y_copy != nullptr) {
      y_copy[i] = convert::To<CopyT>(y[i]);
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT) \
  template <>                                  \
  void name<T, CopyT, CPUContext>(             \
      const int N,                             \
      const float lr,                          \
      const float momentum,                    \
      const float alpha,                       \
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
        convert::To<T>(momentum),              \
        convert::To<T>(alpha),                 \
        convert::To<T>(eps),                   \
        convert::To<T>(wd),                    \
        x,                                     \
        g,                                     \
        m,                                     \
        v,                                     \
        y,                                     \
        y_copy);                               \
  }

DEFINE_KERNEL_LAUNCHER(RMSprop, float, float16);
DEFINE_KERNEL_LAUNCHER(RMSprop, float, float);
DEFINE_KERNEL_LAUNCHER(RMSprop, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
