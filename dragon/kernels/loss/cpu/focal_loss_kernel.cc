#include "dragon/kernels/loss/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename TargetT>
void _SigmoidFocalLoss(
    const int N,
    const int S,
    const int C,
    const int start_index,
    const float alpha,
    const float gamma,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask) {
  const auto NxCxS = N * C * S;
  const float alpha1 = alpha;
  const float alpha2 = 1.f - alpha;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, S};
  for (int i = 0; i < NxCxS; ++i) {
    const float lgt = float(input[i]);
    const int tgt = target[index[0] * S + index[2]];
    const float c1 = float(tgt == (index[1] + start_index));
    const float c2 = float((tgt >= 0) & (tgt != (index[1] + start_index)));
    const float p = 1.f / (1.f + std::exp(-lgt));
    const float logp = std::log(std::max(p, FLT_MIN));
    // pos_term: (1 - p)^{gamma} * log(p)
    const float v1 = std::pow(1.f - p, gamma) * logp;
    // neg_term: p^{gamma} * log(1 - p)
    const float v2 = std::pow(p, gamma) *
        (-lgt * (lgt >= 0.f) -
         std::log(1.f + std::exp(lgt - 2.f * lgt * (lgt >= 0.f))));
    loss[i] = -(c1 * v1 * alpha1 + c2 * v2 * alpha2), mask[i] = c1;
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

template <typename InputT, typename TargetT>
void _SigmoidFocalLossGrad(
    const int N,
    const int S,
    const int C,
    const int start_index,
    const float alpha,
    const float gamma,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask) {
  const auto NxCxS = N * C * S;
  const float alpha1 = alpha;
  const float alpha2 = 1.f - alpha;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, S};
  for (int i = 0; i < NxCxS; ++i) {
    const float lgt = float(input[i]);
    const int tgt = target[index[0] * S + index[2]];
    const float c1 = float(tgt == (index[1] + start_index));
    const float c2 = float((tgt >= 0) & (tgt != (index[1] + start_index)));
    const float p = 1.f / (1.f + std::exp(-lgt));
    const float logp = std::log(std::max(p, FLT_MIN));
    // pos_term: (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
    const float v1 = std::pow(1.f - p, gamma) * (1.f - p - gamma * p * logp);
    // neg_term: p^{gamma} * (gamma * (1 - p) * log(1 - p) - p)
    const float v2 = std::pow(p, gamma) *
        ((-lgt * (lgt >= 0.f) -
          std::log(1.f + std::exp(lgt - 2.f * lgt * (lgt >= 0.f)))) *
             (1.f - p) * gamma -
         p);
    dx[i] = -(c1 * v1 * alpha1 + c2 * v2 * alpha2), mask[i] = c1;
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT)                       \
  template <>                                                               \
  void name<InputT, TargetT, CPUContext>(                                   \
      const int N,                                                          \
      const int S,                                                          \
      const int C,                                                          \
      const int start_index,                                                \
      const float alpha,                                                    \
      const float gamma,                                                    \
      const InputT* input,                                                  \
      const TargetT* target,                                                \
      InputT* loss,                                                         \
      InputT* mask,                                                         \
      CPUContext* ctx) {                                                    \
    _##name(N, S, C, start_index, alpha, gamma, input, target, loss, mask); \
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
