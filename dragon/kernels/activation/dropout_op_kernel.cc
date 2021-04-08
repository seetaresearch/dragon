#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                     \
  template <>                                         \
  void Dropout<T, CPUContext>(                        \
      const int N,                                    \
      const float ratio,                              \
      const float scale,                              \
      const T* x,                                     \
      T* y,                                           \
      uint8_t* mask,                                  \
      uint32_t* /* r */,                              \
      CPUContext* ctx) {                              \
    math::RandomBernoulli(N, 1.f - ratio, mask, ctx); \
    math::ApplyMask(N, scale, mask, x, y, ctx);       \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
