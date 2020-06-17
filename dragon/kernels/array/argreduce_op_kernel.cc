#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _ArgMax(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int top_k,
    const T* x,
    int64_t* y) {
  for (int oix = 0; oix < outer_dim; ++oix) {
    for (int iix = 0; iix < inner_dim; ++iix) {
      const T* X = x + (oix * axis_dim * inner_dim + iix);
      const int y_offset = oix * top_k * inner_dim + iix;
      vector<pair<T, int64_t>> vec(axis_dim);
      for (int j = 0; j < axis_dim; ++j)
        vec[j] = std::make_pair(X[j * inner_dim], j);
      std::partial_sort(
          vec.begin(),
          vec.begin() + top_k,
          vec.end(),
          std::greater<pair<T, int64_t>>());
      for (int j = 0; j < top_k; ++j) {
        y[y_offset + j * inner_dim] = vec[j].second;
      }
    }
  }
}

template <typename T>
void _ArgMin(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int top_k,
    const T* x,
    int64_t* y) {
  for (int oix = 0; oix < outer_dim; ++oix) {
    for (int iix = 0; iix < inner_dim; ++iix) {
      const T* X = x + (oix * axis_dim * inner_dim + iix);
      const int y_offset = oix * top_k * inner_dim + iix;
      vector<pair<T, int64_t>> vec(axis_dim);
      for (int j = 0; j < axis_dim; ++j)
        vec[j] = std::make_pair(X[j * inner_dim], j);
      std::partial_sort(vec.begin(), vec.begin() + top_k, vec.end());
      for (int j = 0; j < top_k; ++j) {
        y[y_offset + j * inner_dim] = vec[j].second;
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
    const int top_k,
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
    const int top_k,
    const float16* x,
    int64_t* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(name, T)                   \
  template <>                                             \
  void name<T, CPUContext>(                               \
      const int outer_dim,                                \
      const int inner_dim,                                \
      const int axis_dim,                                 \
      const int top_k,                                    \
      const T* x,                                         \
      int64_t* y,                                         \
      CPUContext* ctx) {                                  \
    _##name(outer_dim, inner_dim, axis_dim, top_k, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(ArgMax, bool);
DEFINE_KERNEL_LAUNCHER(ArgMax, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, int);
DEFINE_KERNEL_LAUNCHER(ArgMax, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, float);
DEFINE_KERNEL_LAUNCHER(ArgMax, double);
DEFINE_KERNEL_LAUNCHER(ArgMin, bool);
DEFINE_KERNEL_LAUNCHER(ArgMin, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, int);
DEFINE_KERNEL_LAUNCHER(ArgMin, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, float);
DEFINE_KERNEL_LAUNCHER(ArgMin, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
