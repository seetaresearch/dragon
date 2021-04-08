#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _ChannelAffine(
    const int N,
    const int S,
    const int C,
    const T* x,
    const T* scale,
    const T* bias,
    T* y) {
  if (S == 1) {
    if (bias != nullptr) {
      EigenArrayMap<T>(y, C, N) = (ConstEigenArrayMap<T>(x, C, N).colwise() *
                                   ConstEigenVectorArrayMap<T>(scale, C))
                                      .colwise() +
          ConstEigenVectorArrayMap<T>(bias, C);
    } else {
      EigenArrayMap<T>(y, C, N) = ConstEigenArrayMap<T>(x, C, N).colwise() *
          ConstEigenVectorArrayMap<T>(scale, C);
    }
    return;
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      if (bias != nullptr) {
        EigenVectorArrayMap<T>(y, S) =
            ConstEigenVectorArrayMap<T>(x, S) * scale[j] + bias[j];
      } else {
        EigenVectorArrayMap<T>(y, S) =
            ConstEigenVectorArrayMap<T>(x, S) * scale[j];
      }
      x += S;
      y += S;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ChannelAffine<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const float16* x,
    const float16* w,
    const float16* b,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)               \
  template <>                                   \
  void ChannelAffine<T, CPUContext>(            \
      const int N,                              \
      const int S,                              \
      const int C,                              \
      const T* x,                               \
      const T* scale,                           \
      const T* bias,                            \
      T* y,                                     \
      CPUContext* ctx) {                        \
    _ChannelAffine(N, S, C, x, scale, bias, y); \
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
