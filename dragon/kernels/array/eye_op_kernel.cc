#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SetEye(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    T* y) {
  const auto MxN = M * N;
  if (k > 0) {
    const int imax = std::min(M, N - k);
    if (imax <= 0) return;
    for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
      for (int i = 0; i < imax; ++i) {
        y[i * N + k + i] = convert::To<T>(1.f);
      }
      y += MxN;
    }
  } else if (k <= 0) {
    const int imax = std::min(M + k, N);
    if (imax <= 0) return;
    for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
      T* offset_y = y - k * N;
      for (int i = 0; i < imax; ++i) {
        offset_y[i * N + i] = convert::To<T>(1.f);
      }
      y += MxN;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                             \
  template <>                                                 \
  void SetEye<T, CPUContext>(                                 \
      const int batch_size,                                   \
      const int M,                                            \
      const int N,                                            \
      const int k,                                            \
      T* y,                                                   \
      CPUContext* ctx) {                                      \
    math::Set(batch_size* M* N, convert::To<T>(0.f), y, ctx); \
    _SetEye(batch_size, M, N, k, y);                          \
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
