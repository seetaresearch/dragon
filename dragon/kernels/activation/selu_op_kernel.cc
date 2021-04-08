#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
void _Selu(const int N, const AccT alpha, const AccT gamma, const T* x, T* y) {
  const AccT scale = alpha * gamma;
  for (int i = 0; i < N; ++i) {
    const AccT val = convert::To<AccT>(x[i]);
    y[i] = convert::To<T>(
        val > AccT(0) ? gamma * val
                      : scale * ((std::exp(std::min(val, AccT(0))) - AccT(1))));
  }
}

template <typename T, typename AccT>
void _SeluGrad(
    const int N,
    const AccT alpha,
    const AccT gamma,
    const T* dy,
    const T* y,
    T* dx) {
  const AccT scale = alpha * gamma;
  for (int i = 0; i < N; ++i) {
    const AccT val = convert::To<AccT>(y[i]);
    const AccT grad = convert::To<AccT>(dy[i]);
    dx[i] = convert::To<T>(grad * (val > AccT(0) ? gamma : (scale + val)));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                          \
  template <>                                              \
  void Selu<T, CPUContext>(                                \
      const int N,                                         \
      const float alpha,                                   \
      const float gamma,                                   \
      const T* x,                                          \
      T* y,                                                \
      CPUContext* ctx) {                                   \
    _Selu(                                                 \
        N,                                                 \
        convert::To<math::AccmulatorType<T>::type>(alpha), \
        convert::To<math::AccmulatorType<T>::type>(gamma), \
        x,                                                 \
        y);                                                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                     \
  template <>                                              \
  void SeluGrad<T, CPUContext>(                            \
      const int N,                                         \
      const float alpha,                                   \
      const float gamma,                                   \
      const T* dy,                                         \
      const T* y,                                          \
      T* dx,                                               \
      CPUContext* ctx) {                                   \
    _SeluGrad(                                             \
        N,                                                 \
        convert::To<math::AccmulatorType<T>::type>(alpha), \
        convert::To<math::AccmulatorType<T>::type>(gamma), \
        dy,                                                \
        y,                                                 \
        dx);                                               \
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
