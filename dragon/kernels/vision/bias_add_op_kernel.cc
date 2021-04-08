#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _BiasAdd(
    const int N,
    const int S,
    const int C,
    const T* x,
    const T* bias,
    T* y) {
  if (S == 1) {
    EigenArrayMap<T>(y, C, N) = ConstEigenArrayMap<T>(x, C, N).colwise() +
        ConstEigenVectorArrayMap<T>(bias, C);
    return;
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      EigenVectorArrayMap<T>(y, S) =
          ConstEigenVectorArrayMap<T>(x, S) + bias[j];
      x += S;
      y += S;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void BiasAdd<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const float16* x,
    const float16* bias,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)  \
  template <>                      \
  void BiasAdd<T, CPUContext>(     \
      const int N,                 \
      const int S,                 \
      const int C,                 \
      const T* x,                  \
      const T* bias,               \
      T* y,                        \
      CPUContext* ctx) {           \
    _BiasAdd(N, S, C, x, bias, y); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
