#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _ArgMax(const int N, const int S, const int C, const T* x, int64_t* y) {
  using AccT = typename math::Traits<T>::accumulator_type;
  const auto Functor = std::greater<pair<AccT, int64_t>>();
  for (int i = 0; i < N; ++i) {
    vector<pair<AccT, int64_t>> X(C);
    for (int j = 0; j < S; ++j) {
      auto* offset_x = x + i * C * S + j;
      auto* offset_y = y + i * S + j;
      for (int k = 0; k < C; ++k) {
        X[k] = std::make_pair(convert::To<AccT>(offset_x[k * S]), k);
      }
      std::partial_sort(X.begin(), X.begin() + 1, X.end(), Functor);
      *offset_y = X[0].second;
    }
  }
}

template <typename T>
void _ArgMin(const int N, const int S, const int C, const T* x, int64_t* y) {
  using AccT = typename math::Traits<T>::accumulator_type;
  const auto Functor = std::less<pair<AccT, int64_t>>();
  for (int i = 0; i < N; ++i) {
    vector<pair<AccT, int64_t>> X(C);
    for (int j = 0; j < S; ++j) {
      auto* offset_x = x + i * C * S + j;
      auto* offset_y = y + i * S + j;
      for (int k = 0; k < C; ++k) {
        X[k] = std::make_pair(convert::To<AccT>(offset_x[k * S]), k);
      }
      std::partial_sort(X.begin(), X.begin() + 1, X.end(), Functor);
      *offset_y = X[0].second;
    }
  }
}

} // namespace

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
DEFINE_KERNEL_LAUNCHER(ArgMax, float16);
DEFINE_KERNEL_LAUNCHER(ArgMax, bfloat16);
DEFINE_KERNEL_LAUNCHER(ArgMax, float);
DEFINE_KERNEL_LAUNCHER(ArgMax, double);
DEFINE_KERNEL_LAUNCHER(ArgMin, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, int);
DEFINE_KERNEL_LAUNCHER(ArgMin, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, float16);
DEFINE_KERNEL_LAUNCHER(ArgMin, bfloat16);
DEFINE_KERNEL_LAUNCHER(ArgMin, float);
DEFINE_KERNEL_LAUNCHER(ArgMin, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
