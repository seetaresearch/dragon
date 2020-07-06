#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _CumSum(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const bool exclusive,
    const T* x,
    T* y,
    CPUContext* ctx) {
  const int count = outer_dim * axis_dim * inner_dim;
  std::array<int, 3> idx = {0, 0, 0};
  std::array<int, 3> dims = {outer_dim, axis_dim, inner_dim};
  for (int i = 0; i < count; ++i) {
    if (idx[1] > 0) {
      const int j = i - inner_dim;
      y[i] = y[j] + x[exclusive ? j : i];
    } else {
      y[i] = exclusive ? T(0) : x[i];
    }
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <>
void _CumSum<float16>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const bool exclusive,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <typename T>
void _CumSumReverse(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const bool exclusive,
    const T* x,
    T* y,
    CPUContext* ctx) {
  const int kStart = axis_dim - 1;
  for (int n = 0; n < outer_dim; ++n) {
    const int n_offset = n * axis_dim;
    for (int m = kStart; m >= 0; --m) {
      const int nm_offset = (n_offset + m) * inner_dim;
      for (int k = 0; k < inner_dim; ++k) {
        const int i = nm_offset + k;
        if (m < kStart) {
          const int j = i + inner_dim;
          y[i] = y[j] + x[exclusive ? j : i];
        } else {
          y[i] = exclusive ? T(0) : x[i];
        }
      } // End k
    } // End m
  } // End n
}

template <>
void _CumSumReverse<float16>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const bool exclusive,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void CumSum<T, CPUContext>(                                               \
      const int outer_dim,                                                  \
      const int axis_dim,                                                   \
      const int inner_dim,                                                  \
      const bool exclusive,                                                 \
      const bool reverse,                                                   \
      const T* x,                                                           \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    if (reverse) {                                                          \
      _CumSumReverse(outer_dim, axis_dim, inner_dim, exclusive, x, y, ctx); \
    } else {                                                                \
      _CumSum(outer_dim, axis_dim, inner_dim, exclusive, x, y, ctx);        \
    }                                                                       \
  }

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
