#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _ChannelShuffle(
    const int outer_dim,
    const int inner_dim,
    const int G,
    const int K,
    const T* x,
    T* y) {
  for (int i = 0; i < outer_dim; ++i) {
    for (int gi = 0; gi < G; ++gi) {
      for (int ki = 0; ki < K; ++ki) {
        std::memcpy(
            y + ((i * K + ki) * G + gi) * inner_dim,
            x + ((i * G + gi) * K + ki) * inner_dim,
            inner_dim * sizeof(T));
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void ChannelShuffle<T, CPUContext>(                                     \
      const int outer_dim,                                                \
      const int inner_dim,                                                \
      const int axis_dim,                                                 \
      const int group,                                                    \
      const T* x,                                                         \
      T* y,                                                               \
      CPUContext* ctx) {                                                  \
    _ChannelShuffle(outer_dim, inner_dim, group, axis_dim / group, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(bool);
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
