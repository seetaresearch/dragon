#include "dragon/kernels/training/op_kernels.h"
#include "dragon/utils/conversions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename CopyT>
void _MomentumSGD(
    const int N,
    const T lr,
    const T momentum,
    const T wd,
    const T* x,
    const T* g,
    T* m,
    T* y,
    CopyT* y_copy) {
  for (int i = 0; i < N; ++i) {
    const T gi = wd > T(0) ? std::fma(wd, x[i], g[i]) : g[i];
    const T mi = m[i] = std::fma(momentum, m[i], gi);
    y[i] -= lr * mi;
    if (y_copy != nullptr) y_copy[i] = convert::To<CopyT>(y[i]);
  }
}

template <typename T, typename CopyT>
void _NesterovSGD(
    const int N,
    const T lr,
    const T momentum,
    const T wd,
    const T* x,
    const T* g,
    T* m,
    T* y,
    CopyT* y_copy) {
  for (int i = 0; i < N; ++i) {
    const T gi = wd > T(0) ? std::fma(wd, x[i], g[i]) : g[i];
    const T mi = m[i] = std::fma(momentum, m[i], gi);
    y[i] -= lr * std::fma(momentum, mi, gi);
    if (y_copy != nullptr) y_copy[i] = convert::To<CopyT>(y[i]);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, CopyT) \
  template <>                                  \
  void name<T, CopyT, CPUContext>(             \
      const int N,                             \
      const float lr,                          \
      const float momentum,                    \
      const float wd,                          \
      const T* x,                              \
      const T* g,                              \
      T* m,                                    \
      T* y,                                    \
      CopyT* y_copy,                           \
      CPUContext* ctx) {                       \
    _##name(                                   \
        N,                                     \
        convert::To<T>(lr),                    \
        convert::To<T>(momentum),              \
        convert::To<T>(wd),                    \
        x,                                     \
        g,                                     \
        m,                                     \
        y,                                     \
        y_copy);                               \
  }

DEFINE_KERNEL_LAUNCHER(MomentumSGD, float, float16);
DEFINE_KERNEL_LAUNCHER(MomentumSGD, float, bfloat16);
DEFINE_KERNEL_LAUNCHER(MomentumSGD, float, float);
DEFINE_KERNEL_LAUNCHER(MomentumSGD, double, double);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, float, float16);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, float, bfloat16);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, float, float);
DEFINE_KERNEL_LAUNCHER(NesterovSGD, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
