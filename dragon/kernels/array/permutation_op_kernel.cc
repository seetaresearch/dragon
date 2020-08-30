#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _SwapByKey(const int count, const uint32_t* r, T* y) {
  for (int i = 0; i < count; ++i) {
    std::swap(y[i], y[i + (r[i] % (count - i))]);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                            \
  template <>                                                \
  void Permutation<T, CPUContext>(                           \
      const int count, T* y, uint32_t* r, CPUContext* ctx) { \
    math::Random(count, r, ctx);                             \
    kernel::Range(count, 0.f, 1.f, y, ctx);                  \
    _SwapByKey(count, r, y);                                 \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
