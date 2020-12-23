#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename LogitT, typename TargetT>
void _SigmoidFocalLoss(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const LogitT pos_alpha,
    const LogitT neg_alpha,
    const LogitT gamma,
    const int negative_index,
    const LogitT* logit,
    const TargetT* target,
    LogitT* loss,
    LogitT* mask) {
  std::array<int, 3> idx = {0, 0, 0};
  std::array<int, 3> dims = {outer_dim, axis_dim, inner_dim};
  const int count = dims[0] * dims[1] * dims[2];

  for (int i = 0; i < count; ++i) {
    const int t = (int)target[idx[0] * inner_dim + idx[2]];
    // "0" is reserved for target if negative index is zero
    LogitT c1 = (LogitT)(t == (idx[1] + (negative_index ? 0 : 1)));
    LogitT c2 = (LogitT)((t >= 0) & (t != (idx[1] + (negative_index ? 0 : 1))));
    LogitT p = LogitT(1) / (LogitT(1) + std::exp(-logit[i]));

    // (1 - p)^{gamma} * log(p)
    LogitT pos_term =
        std::pow(LogitT(1) - p, gamma) * std::log(std::max(p, (LogitT)FLT_MIN));

    // p^{gamma} * log(1 - p)
    LogitT neg_term = std::pow(p, gamma) *
        (-logit[i] * (logit[i] >= 0) -
         std::log(
             LogitT(1) + std::exp(logit[i] - 2 * logit[i] * (logit[i] >= 0))));

    loss[i] = LogitT(0);
    loss[i] += -c1 * pos_term * pos_alpha;
    loss[i] += -c2 * neg_term * neg_alpha;
    mask[i] = c1;

    math::utils::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename LogitT, typename TargetT>
void _SigmoidFocalLossGrad(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const LogitT pos_alpha,
    const LogitT neg_alpha,
    const LogitT gamma,
    const int negative_index,
    const LogitT* logit,
    const TargetT* target,
    LogitT* dx,
    LogitT* mask) {
  std::array<int, 3> idx = {0, 0, 0};
  std::array<int, 3> dims = {outer_dim, axis_dim, inner_dim};
  const int count = dims[0] * dims[1] * dims[2];

  for (int i = 0; i < count; ++i) {
    const int t = (int)target[idx[0] * inner_dim + idx[2]];
    // "0" is reserved for target if negative index is zero
    LogitT c1 = (LogitT)(t == (idx[1] + (negative_index ? 0 : 1)));
    LogitT c2 = (LogitT)((t >= 0) & (t != (idx[1] + (negative_index ? 0 : 1))));
    LogitT p = LogitT(1) / (LogitT(1) + std::exp(-logit[i]));

    // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
    LogitT pos_term = std::pow(LogitT(1) - p, gamma) *
        (LogitT(1) - p - p * gamma * std::log(std::max(p, (LogitT)FLT_MIN)));

    // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
    LogitT neg_term = std::pow(p, gamma) *
        ((-logit[i] * (logit[i] >= 0) -
          std::log(
              LogitT(1) +
              std::exp(logit[i] - LogitT(2) * logit[i] * (logit[i] >= 0)))) *
             (1 - p) * gamma -
         p);

    dx[i] = LogitT(0);
    dx[i] += -c1 * pos_term * pos_alpha;
    dx[i] += -c2 * neg_term * neg_alpha;
    mask[i] = c1;

    math::utils::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, LogitT, TargetT) \
  template <>                                         \
  void name<LogitT, TargetT, CPUContext>(             \
      const int outer_dim,                            \
      const int inner_dim,                            \
      const int axis_dim,                             \
      const float pos_alpha,                          \
      const float neg_alpha,                          \
      const float gamma,                              \
      const int negative_index,                       \
      const LogitT* logit,                            \
      const TargetT* target,                          \
      LogitT* loss,                                   \
      LogitT* mask,                                   \
      CPUContext* ctx) {                              \
    _##name(                                          \
        outer_dim,                                    \
        inner_dim,                                    \
        axis_dim,                                     \
        (LogitT)pos_alpha,                            \
        (LogitT)neg_alpha,                            \
        (LogitT)gamma,                                \
        negative_index,                               \
        logit,                                        \
        target,                                       \
        loss,                                         \
        mask);                                        \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, float);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, double);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, float);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, double);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
