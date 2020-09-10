#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Unique(
    const int dim,
    const T* x,
    T* y,
    int64_t* inverse_index,
    int64_t* counts,
    int* num) {
  vec32_t order(dim);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [x](const int i, const int j) {
    return x[i] < x[j];
  });
  int n = dim, m;
  for (int i = 1; i < dim; ++i) {
    n -= x[order[i]] == x[order[i - 1]];
  }
  n = 0;
  T prev = -1;
  for (int i = 0; i < dim; ++i) {
    if (i == 0 || prev != x[order[i]]) {
      if (counts && i > 0) counts[n - 1] = m;
      prev = y[n++] = x[order[i]];
      m = 1;
    } else {
      m += 1;
    }
    if (inverse_index) {
      inverse_index[order[i]] = n - 1;
    }
  }
  num[0] = n;
  if (counts) counts[n - 1] = m;
}

} // namespace

template <>
void Unique<float16, CPUContext>(
    const int dim,
    const float16* x,
    float16* y,
    int64_t* inverse_index,
    int64_t* counts,
    int* num,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)                   \
  template <>                                       \
  void Unique<T, CPUContext>(                       \
      const int dim,                                \
      const T* x,                                   \
      T* y,                                         \
      int64_t* inverse_index,                       \
      int64_t* counts,                              \
      int* num,                                     \
      CPUContext* ctx) {                            \
    _Unique(dim, x, y, inverse_index, counts, num); \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

} // namespace kernel

} // namespace dragon
