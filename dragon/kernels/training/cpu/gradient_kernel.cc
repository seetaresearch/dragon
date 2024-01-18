#include "dragon/kernels/training/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _CheckFinite(const int N, const T* g, float* isinf) {
  for (int i = 0; i < N; ++i) {
    if (!math::utils::IsFinite(g[i])) {
      *isinf = 1.f;
      break;
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                         \
  template <>                                                   \
  void name<T, CPUContext>(                                     \
      const int N, const T* g, float* isinf, CPUContext* ctx) { \
    _##name(N, g, isinf);                                       \
  }

DEFINE_KERNEL_LAUNCHER(CheckFinite, float16);
DEFINE_KERNEL_LAUNCHER(CheckFinite, bfloat16);
DEFINE_KERNEL_LAUNCHER(CheckFinite, float);
DEFINE_KERNEL_LAUNCHER(CheckFinite, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
