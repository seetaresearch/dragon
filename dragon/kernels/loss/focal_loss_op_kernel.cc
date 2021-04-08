#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename TargetT>
void _SigmoidFocalLoss(
    const int N,
    const int S,
    const int C,
    const int start_index,
    const InputT alpha,
    const InputT gamma,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask) {
  const auto NxCxS = N * C * S;
  const auto pos_alpha = alpha;
  const auto neg_alpha = InputT(1) - alpha;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, S};
  for (int i = 0; i < NxCxS; ++i) {
    const auto t = target[index[0] * S + index[2]];
    InputT c1 = InputT(t == (index[1] + start_index));
    InputT c2 = InputT((t >= 0) & (t != (index[1] + start_index)));
    InputT p = InputT(1) / (InputT(1) + std::exp(-input[i]));
    InputT logp = std::log(std::max(p, InputT(FLT_MIN)));
    // (1 - p)^{gamma} * log(p)
    InputT pos_term = std::pow(InputT(1) - p, gamma) * logp;
    // p^{gamma} * log(1 - p)
    InputT neg_term = std::pow(p, gamma) *
        (-input[i] * (input[i] >= InputT(0)) -
         std::log(
             InputT(1) +
             std::exp(input[i] - 2 * input[i] * (input[i] >= InputT(0)))));
    loss[i] = InputT(0);
    loss[i] += -c1 * pos_term * pos_alpha;
    loss[i] += -c2 * neg_term * neg_alpha;
    mask[i] = c1;
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

template <typename InputT, typename TargetT>
void _SigmoidFocalLossGrad(
    const int N,
    const int S,
    const int C,
    const int start_index,
    const InputT alpha,
    const InputT gamma,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask) {
  const auto NxCxS = N * C * S;
  const auto pos_alpha = alpha;
  const auto neg_alpha = InputT(1) - alpha;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, S};
  for (int i = 0; i < NxCxS; ++i) {
    const auto t = target[index[0] * S + index[2]];
    InputT c1 = InputT(t == (index[1] + start_index));
    InputT c2 = InputT((t >= 0) & (t != (index[1] + start_index)));
    InputT p = InputT(1) / (InputT(1) + std::exp(-input[i]));
    InputT logp = std::log(std::max(p, InputT(FLT_MIN)));
    // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
    InputT pos_term =
        std::pow(InputT(1) - p, gamma) * (InputT(1) - p - gamma * p * logp);
    // p^{gamma} * (gamma * (1 - p) * log(1 - p) - p)
    InputT neg_term = std::pow(p, gamma) *
        ((-input[i] * (input[i] >= InputT(0)) -
          std::log(
              InputT(1) +
              std::exp(
                  input[i] - InputT(2) * input[i] * (input[i] >= InputT(0))))) *
             (InputT(1) - p) * gamma -
         p);
    dx[i] = InputT(0);
    dx[i] += -c1 * pos_term * pos_alpha;
    dx[i] += -c2 * neg_term * neg_alpha;
    mask[i] = c1;
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT) \
  template <>                                         \
  void name<InputT, TargetT, CPUContext>(             \
      const int N,                                    \
      const int S,                                    \
      const int C,                                    \
      const int start_index,                          \
      const float alpha,                              \
      const float gamma,                              \
      const InputT* input,                            \
      const TargetT* target,                          \
      InputT* loss,                                   \
      InputT* mask,                                   \
      CPUContext* ctx) {                              \
    _##name(                                          \
        N,                                            \
        S,                                            \
        C,                                            \
        start_index,                                  \
        InputT(alpha),                                \
        InputT(gamma),                                \
        input,                                        \
        target,                                       \
        loss,                                         \
        mask);                                        \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
