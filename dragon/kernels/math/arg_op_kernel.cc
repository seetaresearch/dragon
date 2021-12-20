#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _ArgMax(const int N, const int S, const int C, const T* x, int64_t* y) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      auto* offset_x = x + i * C * S + j;
      auto* offset_y = y + i * S + j;
      vector<pair<T, int64_t>> vec(C);
      for (int k = 0; k < C; ++k) {
        vec[k] = std::make_pair(offset_x[k * S], k);
      }
      std::partial_sort(
          vec.begin(),
          vec.begin() + 1,
          vec.end(),
          std::greater<pair<T, int64_t>>());
      *offset_y = vec[0].second;
    }
  }
}

template <typename T>
void _ArgMin(const int N, const int S, const int C, const T* x, int64_t* y) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      auto* offset_x = x + i * C * S + j;
      auto* offset_y = y + i * S + j;
      vector<pair<T, int64_t>> vec(C);
      for (int k = 0; k < C; ++k) {
        vec[k] = std::make_pair(offset_x[k * S], k);
      }
      std::partial_sort(vec.begin(), vec.begin() + 1, vec.end());
      *offset_y = vec[0].second;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ArgMax<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const float16* x,
    int64_t* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void ArgMin<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const float16* x,
    int64_t* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CPUContext>(             \
      const int N,                      \
      const int S,                      \
      const int C,                      \
      const T* x,                       \
      int64_t* y,                       \
      CPUContext* ctx) {                \
    _##name(N, S, C, x, y);             \
  }

DEFINE_KERNEL_LAUNCHER(ArgMax, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, int);
DEFINE_KERNEL_LAUNCHER(ArgMax, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, float);
DEFINE_KERNEL_LAUNCHER(ArgMax, double);
DEFINE_KERNEL_LAUNCHER(ArgMin, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, int);
DEFINE_KERNEL_LAUNCHER(ArgMin, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, float);
DEFINE_KERNEL_LAUNCHER(ArgMin, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
