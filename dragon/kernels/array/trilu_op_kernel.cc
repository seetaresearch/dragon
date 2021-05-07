#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SetTrilu(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    const int upper,
    T* y) {
  if (upper) {
    for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
      for (int i = 0; i < M; ++i) {
        const int jmax = std::min(i + k, N);
        for (int j = 0; j < jmax; ++j) {
          y[j] = convert::To<T>(0.f);
        }
        y += N;
      }
    }
  } else {
    for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
      for (int i = 0; i < M; ++i) {
        const int jmin = std::max(i + k + 1, 0);
        for (int j = jmin; j < N; ++j) {
          y[j] = convert::To<T>(0.f);
        }
        y += N;
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)             \
  template <>                                 \
  void SetTrilu<T, CPUContext>(               \
      const int batch_size,                   \
      const int M,                            \
      const int N,                            \
      const int k,                            \
      const int upper,                        \
      const T* x,                             \
      T* y,                                   \
      CPUContext* ctx) {                      \
    math::Copy(batch_size* M* N, x, y, ctx);  \
    _SetTrilu(batch_size, M, N, k, upper, y); \
  }

DEFINE_KERNEL_LAUNCHER(bool);
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
