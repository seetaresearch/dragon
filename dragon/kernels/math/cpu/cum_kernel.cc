#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, class Reducer>
void _CumReduce(
    const int N,
    const int S,
    const int C,
    const T init,
    const bool exclusive,
    const T* x,
    T* y) {
  const auto reducer = Reducer();
  const auto NxCxS = N * C * S;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, S};
  for (int i = 0; i < NxCxS; ++i) {
    if (index[1] > 0) {
      const int offset = i - S;
      y[i] = reducer(y[offset], x[exclusive ? offset : i]);
    } else {
      y[i] = exclusive ? init : x[i];
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

template <typename T, class Reducer>
void _CumRevReduce(
    const int N,
    const int S,
    const int C,
    const T init,
    const bool exclusive,
    const T* x,
    T* y) {
  const auto reducer = Reducer();
  for (int i = 0; i < N; ++i) {
    for (int k = C - 1; k >= 0; --k) {
      for (int j = 0; j < S; ++j) {
        const int index = (i * C + k) * S + j;
        if (k < C - 1) {
          const int offset = index + S;
          y[index] = reducer(y[offset], x[exclusive ? offset : index]);
        } else {
          y[index] = exclusive ? init : x[index];
        }
      }
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T, Reducer, kInit)                        \
  template <>                                                                  \
  void name<T, CPUContext>(                                                    \
      const int N,                                                             \
      const int S,                                                             \
      const int C,                                                             \
      const bool exclusive,                                                    \
      const bool reverse,                                                      \
      const T* x,                                                              \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    const T init = convert::To<T>(kInit);                                      \
    if (!reverse) _CumReduce<T, Reducer<T>>(N, S, C, init, exclusive, x, y);   \
    if (reverse) _CumRevReduce<T, Reducer<T>>(N, S, C, init, exclusive, x, y); \
  }

// clang-format off
DEFINE_KERNEL_LAUNCHER(CumSum, uint8_t, math::PlusFunctor, uint8_t(0));
DEFINE_KERNEL_LAUNCHER(CumSum, int8_t, math::PlusFunctor, int8_t(0));
DEFINE_KERNEL_LAUNCHER(CumSum, int, math::PlusFunctor, int(0));
DEFINE_KERNEL_LAUNCHER(CumSum, int64_t, math::PlusFunctor, int64_t(0));
DEFINE_KERNEL_LAUNCHER(CumSum, float16, math::PlusFunctor, 0.f);
DEFINE_KERNEL_LAUNCHER(CumSum, bfloat16, math::PlusFunctor, 0.f);
DEFINE_KERNEL_LAUNCHER(CumSum, float, math::PlusFunctor, 0.f);
DEFINE_KERNEL_LAUNCHER(CumSum, double, math::PlusFunctor, 0.);
DEFINE_KERNEL_LAUNCHER(CumMax, uint8_t, math::MaxFunctor, math::Traits<uint8_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, int8_t, math::MaxFunctor, math::Traits<int8_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, int, math::MaxFunctor, math::Traits<int>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, int64_t, math::MaxFunctor, math::Traits<int64_t>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, float16, math::MaxFunctor, math::Traits<float16>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, bfloat16, math::MaxFunctor, math::Traits<bfloat16>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, float, math::MaxFunctor, math::Traits<float>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMax, double, math::MaxFunctor, math::Traits<double>::Lowest());
DEFINE_KERNEL_LAUNCHER(CumMin, uint8_t, math::MinFunctor, math::Traits<uint8_t>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, int8_t, math::MinFunctor, math::Traits<int8_t>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, int, math::MinFunctor, math::Traits<int>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, int64_t, math::MinFunctor, math::Traits<int64_t>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, float16, math::MinFunctor, math::Traits<float16>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, bfloat16, math::MinFunctor, math::Traits<bfloat16>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, float, math::MinFunctor, math::Traits<float>::Max());
DEFINE_KERNEL_LAUNCHER(CumMin, double, math::MinFunctor, math::Traits<double>::Max());
#undef DEFINE_KERNEL_LAUNCHER // clang-format on

} // namespace kernels

} // namespace dragon
