#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename LogitType, typename TargetType>
void _SparseSoftmaxCrossEntropy(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* prob,
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
      loss[i] = -std::log(std::max(prob[k], LogitType(FLT_MIN)));
      mask[i] = LogitType(1);
    }
    utils::math::IncreaseIndexInDims(2, dims.data(), idx.data());
  }
}

template <typename LogitType, typename TargetType>
void _SparseSoftmaxCrossEntropyGrad(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* prob,
    const TargetType* target,
    LogitType* dx,
    LogitType* mask) {
  std::array<int, 2> idx = {0, 0};
  std::array<int, 2> dims = {outer_dim, inner_dim};
  int count = dims[0] * dims[1], k;
  for (int i = 0; i < count; ++i) {
    const int label = (int)target[i];
    if (label == ignore_index) {
      LogitType* offset_dx = dx + idx[0] * axis_dim * inner_dim + idx[1];
      for (int j = 0; j < axis_dim; ++j) {
        (*offset_dx) = LogitType(0);
        offset_dx += inner_dim;
      }
      mask[i] = LogitType(0);
    } else {
      k = (idx[0] * axis_dim + label) * inner_dim + idx[1];
      dx[k] -= LogitType(1);
      mask[i] = LogitType(1);
    }
    utils::math::IncreaseIndexInDims(2, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, LogitType, TargetType) \
  template <>                                               \
  void name<LogitType, TargetType, CPUContext>(             \
      const int outer_dim,                                  \
      const int axis_dim,                                   \
      const int inner_dim,                                  \
      const int ignore_index,                               \
      const LogitType* prob,                                \
      const TargetType* target,                             \
      LogitType* loss,                                      \
      LogitType* mask,                                      \
      CPUContext* ctx) {                                    \
    _##name(                                                \
        outer_dim,                                          \
        axis_dim,                                           \
        inner_dim,                                          \
        ignore_index,                                       \
        prob,                                               \
        target,                                             \
        loss,                                               \
        mask);                                              \
  }

DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, float, float);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, double, double);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, double, int64_t);

DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, float, float);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, double, double);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, double, int64_t);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
