#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename TargetT>
void _NLLLoss(
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
      loss[i] = -input[(index[0] * C + t) * S + index[1]];
      mask[i] = InputT(1);
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), index.data());
  }
}

template <typename InputT, typename TargetT>
void _NLLLossGrad(
    const int N,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask) {
  const auto NxS = N * S;
  std::array<int, 2> index = {0, 0};
  std::array<int, 2> dims = {N, S};
  for (int i = 0; i < NxS; ++i) {
    const auto t = int(target[i]);
    if (t == ignore_index) {
      mask[i] = InputT(0);
    } else {
      dx[(index[0] * C + t) * S + index[1]] = InputT(-1);
      mask[i] = InputT(1);
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

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

DEFINE_KERNEL_LAUNCHER(NLLLoss, float, int);
DEFINE_KERNEL_LAUNCHER(NLLLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, int);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, int);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, int);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
