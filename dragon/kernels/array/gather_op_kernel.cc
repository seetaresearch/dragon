#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Gather(
    const int N,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* x,
    T* y,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < K; ++k) {
      int pos = index[k];
      pos = (pos >= 0 ? pos : pos + C);
      const T* offset_x = x + (i * C + pos) * S;
      math::Copy(S, offset_x, y, ctx);
      y += S;
    }
  }
}

template <typename T>
void _GatherGrad(
    const int N,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* dy,
    float* dx,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < K; ++k) {
      int pos = index[k];
      pos = (pos >= 0 ? pos : pos + C);
      float* offset_dx = dx + (i * C + pos) * S;
      for (int j = 0; j < S; ++j) {
        offset_dx[j] += convert::To<float>(dy[j]);
      }
      dy += S;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, InputT, OutputT) \
  template <>                                         \
  void name<InputT, CPUContext>(                      \
      const int N,                                    \
      const int S,                                    \
      const int C,                                    \
      const int K,                                    \
      const int64_t* index,                           \
      const InputT* x,                                \
      OutputT* y,                                     \
      CPUContext* ctx) {                              \
    _##name(N, S, C, K, index, x, y, ctx);            \
  }

DEFINE_KERNEL_LAUNCHER(Gather, bool, bool);
DEFINE_KERNEL_LAUNCHER(Gather, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(Gather, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(Gather, int, int);
DEFINE_KERNEL_LAUNCHER(Gather, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(Gather, float16, float16);
DEFINE_KERNEL_LAUNCHER(Gather, float, float);
DEFINE_KERNEL_LAUNCHER(Gather, double, double);
DEFINE_KERNEL_LAUNCHER(GatherGrad, float16, float); // GatherGrad
DEFINE_KERNEL_LAUNCHER(GatherGrad, float, float); // GatherGrad
DEFINE_KERNEL_LAUNCHER(GatherGrad, double, float); // GatherGrad
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
