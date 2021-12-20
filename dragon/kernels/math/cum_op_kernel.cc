#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
void _CumSum(
    const int N,
    const int S,
    const int C,
    const bool exclusive,
    const T* x,
    T* y,
    CPUContext* ctx) {
  const auto NxCxS = N * C * S;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, S};
  for (int i = 0; i < NxCxS; ++i) {
    if (index[1] > 0) {
      const int offset = i - S;
      y[i] = convert::To<T>(
          convert::To<AccT>(y[offset]) +
          convert::To<AccT>(x[exclusive ? offset : i]));
    } else {
      y[i] = exclusive ? convert::To<T>(AccT(0)) : x[i];
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
void _CumSumReverse(
    const int N,
    const int S,
    const int C,
    const bool exclusive,
    const T* x,
    T* y,
    CPUContext* ctx) {
  const auto C2 = C - 1;
  for (int i = 0; i < N; ++i) {
    for (int k = C2; k >= 0; --k) {
      for (int j = 0; j < S; ++j) {
        const int index = (i * C + k) * S + j;
        if (k < C2) {
          const int offset = index + S;
          y[index] = convert::To<T>(
              convert::To<AccT>(y[offset]) +
              convert::To<AccT>(x[exclusive ? offset : index]));
        } else {
          y[index] = exclusive ? convert::To<T>(AccT(0)) : x[index];
        }
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                       \
  template <>                                                 \
  void CumSum<T, CPUContext>(                                 \
      const int N,                                            \
      const int S,                                            \
      const int C,                                            \
      const bool exclusive,                                   \
      const bool reverse,                                     \
      const T* x,                                             \
      T* y,                                                   \
      CPUContext* ctx) {                                      \
    if (reverse) {                                            \
      _CumSumReverse<T, AccT>(N, S, C, exclusive, x, y, ctx); \
    } else {                                                  \
      _CumSum<T, AccT>(N, S, C, exclusive, x, y, ctx);        \
    }                                                         \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
