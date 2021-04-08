#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _ChannelShuffle(
    const int N,
    const int S,
    const int G,
    const int K,
    const T* x,
    T* y) {
  for (int i = 0; i < N; ++i) {
    for (int gi = 0; gi < G; ++gi) {
      for (int ki = 0; ki < K; ++ki) {
        std::memcpy(
            y + ((i * K + ki) * G + gi) * S,
            x + ((i * G + gi) * K + ki) * S,
            S * sizeof(T));
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)          \
  template <>                              \
  void ChannelShuffle<T, CPUContext>(      \
      const int N,                         \
      const int S,                         \
      const int C,                         \
      const int G,                         \
      const T* x,                          \
      T* y,                                \
      CPUContext* ctx) {                   \
    _ChannelShuffle(N, S, G, C / G, x, y); \
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
