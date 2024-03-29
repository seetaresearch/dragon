#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SetOneHot(const int N, const int depth, const T value, const T* x, T* y) {
  using AccT = typename math::Traits<T>::accumulator_type;
  for (int i = 0; i < N; ++i) {
    y[i * depth + int(convert::To<AccT>(x[i]))] = value;
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                      \
  template <>                                          \
  void SetOneHot<T, CPUContext>(                       \
      const int N,                                     \
      const int depth,                                 \
      const float value,                               \
      const T* x,                                      \
      T* y,                                            \
      CPUContext* ctx) {                               \
    _SetOneHot(N, depth, convert::To<T>(value), x, y); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
