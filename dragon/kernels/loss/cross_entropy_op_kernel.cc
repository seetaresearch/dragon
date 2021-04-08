#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _CrossEntropy(const int N, const T* input, const T* target, T* loss) {
  for (int i = 0; i < N; ++i) {
    loss[i] = -target[i] * std::log(std::max(input[i], T(FLT_MIN)));
  }
}

template <typename InputT, typename TargetT>
void _CrossEntropy(
    const int N,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask) {
  const auto NxS = N * S;
  std::array<int, 2> index = {0, 0};
  std::array<int, 2> dims = {N, S};
  for (int i = 0; i < NxS; ++i) {
    const auto t = int(target[i]);
    if (t == ignore_index) {
      loss[i] = mask[i] = InputT(0);
    } else {
      const auto j = (index[0] * C + t) * S + index[1];
      loss[i] = -std::log(std::max(input[j], InputT(FLT_MIN)));
      mask[i] = InputT(1);
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), index.data());
  }
}

template <typename T>
void _SigmoidCrossEntropy(
    const int N,
    const T* input,
    const T* target,
    T* loss,
    T* mask) {
  for (int i = 0; i < N; ++i) {
    if (target[i] < 0) {
      loss[i] = mask[i] = T(0);
    } else {
      loss[i] = std::log(
                    T(1) +
                    std::exp(input[i] - T(2) * input[i] * (input[i] >= T(0)))) +
          input[i] * ((input[i] >= T(0)) - target[i]);
      mask[i] = T(1);
    }
  }
}

template <typename T>
void _SigmoidCrossEntropyGrad(
    const int N,
    const T* input,
    const T* target,
    T* dx,
    T* mask) {
  for (int i = 0; i < N; ++i) {
    if (target[i] < 0) {
      dx[i] = mask[i] = T(0);
    } else {
      dx[i] = T(1) / (T(1) + std::exp(-input[i])) - target[i];
      mask[i] = T(1);
    }
  }
}

template <typename InputT, typename TargetT>
void _SoftmaxCrossEntropyGrad(
    const int N,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* grad,
    InputT* mask) {
  const auto NxS = N * S;
  std::array<int, 2> index = {0, 0};
  std::array<int, 2> dims = {N, S};
  for (int i = 0; i < NxS; ++i) {
    const auto t = int(target[i]);
    if (t == ignore_index) {
      InputT* offset_grad = grad + index[0] * C * S + index[1];
      for (int j = 0; j < C; ++j) {
        (*offset_grad) = InputT(0);
        offset_grad += S;
      }
      mask[i] = InputT(0);
    } else {
      const auto j = (index[0] * C + t) * S + index[1];
      grad[j] -= InputT(1);
      mask[i] = InputT(1);
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CPUContext>(             \
      const int N,                      \
      const T* input,                   \
      const T* target,                  \
      T* loss,                          \
      CPUContext* ctx) {                \
    _##name(N, input, target, loss);    \
  }

DEFINE_KERNEL_LAUNCHER(CrossEntropy, float);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)    \
  template <>                              \
  void name<T, CPUContext>(                \
      const int N,                         \
      const T* input,                      \
      const T* target,                     \
      T* loss,                             \
      T* mask,                             \
      CPUContext* ctx) {                   \
    _##name(N, input, target, loss, mask); \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, double);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT)          \
  template <>                                                  \
  void name<InputT, TargetT, CPUContext>(                      \
      const int N,                                             \
      const int S,                                             \
      const int C,                                             \
      const int ignore_index,                                  \
      const InputT* input,                                     \
      const TargetT* target,                                   \
      InputT* loss,                                            \
      InputT* mask,                                            \
      CPUContext* ctx) {                                       \
    _##name(N, S, C, ignore_index, input, target, loss, mask); \
  }

DEFINE_KERNEL_LAUNCHER(CrossEntropy, float, int);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, float, int64_t);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double, int);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, float, int);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, double, int);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
