#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename LogitType, typename TargetType>
void _NLLLoss(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int ignore_index,
    const LogitType* logit,
    const TargetType* target,
    LogitType* loss,
    LogitType* mask) {
  std::array<int, 2> idx = {0, 0};
  std::array<int, 2> dims = {outer_dim, inner_dim};
  int count = dims[0] * dims[1], k;
  for (int i = 0; i < count; ++i) {
    const int label = (int)target[i];
    if (label == ignore_index) {
      loss[i] = mask[i] = LogitType(0);
    } else {
      k = (idx[0] * axis_dim + label) * inner_dim + idx[1];
      loss[i] = -logit[k], mask[i] = LogitType(1);
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), idx.data());
  }
}

template <typename LogitType, typename TargetType>
void _NLLLossGrad(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int ignore_index,
    const LogitType* logit,
    const TargetType* target,
    LogitType* dlogit,
    LogitType* mask) {
  std::array<int, 2> idx = {0, 0};
  std::array<int, 2> dims = {outer_dim, inner_dim};
  int count = dims[0] * dims[1], k;
  for (int i = 0; i < count; ++i) {
    const int label = (int)target[i];
    if (label == ignore_index) {
      mask[i] = LogitType(0);
    } else {
      k = (idx[0] * axis_dim + label) * inner_dim + idx[1];
      dlogit[k] = LogitType(-1), mask[i] = LogitType(1);
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, LogitType, TargetType) \
  template <>                                               \
  void name<LogitType, TargetType, CPUContext>(             \
      const int outer_dim,                                  \
      const int inner_dim,                                  \
      const int axis_dim,                                   \
      const int ignore_index,                               \
      const LogitType* logit,                               \
      const TargetType* target,                             \
      LogitType* loss,                                      \
      LogitType* mask,                                      \
      CPUContext* ctx) {                                    \
    _##name(                                                \
        outer_dim,                                          \
        inner_dim,                                          \
        axis_dim,                                           \
        ignore_index,                                       \
        logit,                                              \
        target,                                             \
        loss,                                               \
        mask);                                              \
  }

DEFINE_KERNEL_LAUNCHER(NLLLoss, float, float);
DEFINE_KERNEL_LAUNCHER(NLLLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, double);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, int64_t);

DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, float);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, double);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, int64_t);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
