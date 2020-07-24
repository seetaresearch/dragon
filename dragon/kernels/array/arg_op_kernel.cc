#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _ArgMax(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    int64_t* y) {
  const int top_k = 1;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      auto* offset_x = x + (i * axis_dim * inner_dim + j);
      auto* offset_y = y + (i * top_k * inner_dim + j);
      vector<pair<T, int64_t>> vec(axis_dim);
      for (int k = 0; k < axis_dim; ++k) {
        vec[k] = std::make_pair(offset_x[k * inner_dim], k);
      }
      std::partial_sort(
          vec.begin(),
          vec.begin() + top_k,
          vec.end(),
          std::greater<pair<T, int64_t>>());
      for (int k = 0; k < top_k; ++k) {
        offset_y[k * inner_dim] = vec[k].second;
      }
    }
  }
}

template <typename T>
void _ArgMin(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    int64_t* y) {
  const int top_k = 1;
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      auto* offset_x = x + (i * axis_dim * inner_dim + j);
      auto* offset_y = y + (i * top_k * inner_dim + j);
      vector<pair<T, int64_t>> vec(axis_dim);
      for (int k = 0; k < axis_dim; ++k) {
        vec[k] = std::make_pair(offset_x[k * inner_dim], k);
      }
      std::partial_sort(vec.begin(), vec.begin() + top_k, vec.end());
      for (int k = 0; k < top_k; ++k) {
        offset_y[k * inner_dim] = vec[k].second;
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ArgMax<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const float16* x,
    int64_t* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void ArgMin<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const float16* x,
    int64_t* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(name, T)            \
  template <>                                      \
  void name<T, CPUContext>(                        \
      const int outer_dim,                         \
      const int inner_dim,                         \
      const int axis_dim,                          \
      const T* x,                                  \
      int64_t* y,                                  \
      CPUContext* ctx) {                           \
    _##name(outer_dim, inner_dim, axis_dim, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(ArgMax, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, int);
DEFINE_KERNEL_LAUNCHER(ArgMax, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, float);
DEFINE_KERNEL_LAUNCHER(ArgMax, double);
DEFINE_KERNEL_LAUNCHER(ArgMin, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, int);
DEFINE_KERNEL_LAUNCHER(ArgMin, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, float);
DEFINE_KERNEL_LAUNCHER(ArgMin, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
