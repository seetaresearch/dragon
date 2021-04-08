#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
void _DropPath(
    const int N,
    const int C,
    const AccT scale,
    const uint8_t* mask,
    const T* x,
    T* y) {
  const auto NxC = N * C;
  for (int i = 0; i < NxC; ++i) {
    y[i] = convert::To<T>(convert::To<AccT>(x[i]) * AccT(mask[i / C]) * scale);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void DropPath<T, CPUContext>(                                        \
      const int N,                                                     \
      const int C,                                                     \
      const float ratio,                                               \
      const float scale,                                               \
      const T* x,                                                      \
      T* y,                                                            \
      uint8_t* mask,                                                   \
      uint32_t* /* r */,                                               \
      CPUContext* ctx) {                                               \
    math::RandomBernoulli(N, 1.f - ratio, mask, ctx);                  \
    _DropPath(N, C, math::AccmulatorType<T>::type(scale), mask, x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                   \
  template <>                                                            \
  void DropPathGrad<T, CPUContext>(                                      \
      const int N,                                                       \
      const int C,                                                       \
      const float scale,                                                 \
      const uint8_t* mask,                                               \
      const T* dy,                                                       \
      T* dx,                                                             \
      CPUContext* ctx) {                                                 \
    _DropPath(N, C, math::AccmulatorType<T>::type(scale), mask, dy, dx); \
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
