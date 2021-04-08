#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SwapByKey(const int N, const uint32_t* r, T* y) {
  for (int i = 0; i < N; ++i) {
    std::swap(y[i], y[i + (r[i] % (N - i))]);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                        \
  template <>                                            \
  void Permutation<T, CPUContext>(                       \
      const int N, T* y, uint32_t* r, CPUContext* ctx) { \
    math::Random(N, r, ctx);                             \
    kernels::Range(N, 0.f, 1.f, y, ctx);                 \
    _SwapByKey(N, r, y);                                 \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
