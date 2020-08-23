#include "dragon/operators/normalization/lp_normalize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define CANONICALIZE_AXES_WITH_TENSOR(tensor)         \
  CANONICALIZE_AXIS_WITH_TENSOR(tensor);              \
  auto num_axes = OpArg<int64_t>("num_axes", 1);      \
  if (num_axes < 0) {                                 \
    num_axes = tensor.ndim() - axis;                  \
  } else if (num_axes == 0) {                         \
    num_axes = 1;                                     \
  }                                                   \
  CHECK(axis + num_axes <= tensor.ndim())             \
      << "\nInvalid number of axes. Got " << num_axes \
      << ", excepted in the range [1, " << tensor.ndim() - axis << "]."

template <class Context>
template <typename T>
void LpNormalizeOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXES_WITH_TENSOR(X);

  // Normalize input with a scaled norm
  auto reduce_dim = X.count(axis, axis + num_axes);
  if (p_ == 1) {
    kernel::L1Normalize(
        X.count(0, axis),
        reduce_dim,
        X.count(axis + num_axes),
        reduction_ == "MEAN" ? 1.f / (float)reduce_dim : 1.f,
        epsilon_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else if (p_ == 2) {
    kernel::L2Normalize(
        X.count(0, axis),
        reduce_dim,
        X.count(axis + num_axes),
        reduction_ == "MEAN" ? 1.f / (float)reduce_dim : 1.f,
        epsilon_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported order of normalization: " << p_;
  }
}

template <class Context>
void LpNormalizeOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void LpNormalizeGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  CANONICALIZE_AXES_WITH_TENSOR(X);

  auto reduce_dim = X.count(axis, axis + num_axes);
  if (p_ == 1) {
    kernel::L1NormalizeGrad(
        X.count(0, axis),
        reduce_dim,
        X.count(axis + num_axes),
        reduction_ == "MEAN" ? 1.f / (float)reduce_dim : 1.f,
        epsilon_,
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else if (p_ == 2) {
    kernel::L2NormalizeGrad(
        X.count(0, axis),
        reduce_dim,
        X.count(axis + num_axes),
        reduction_ == "MEAN" ? 1.f / (float)reduce_dim : 1.f,
        epsilon_,
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported order of LpNormalization: " << p_;
  }
}

template <class Context>
void LpNormalizeGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(LpNormalize);
#ifdef USE_CUDA
DEPLOY_CUDA(LpNormalize);
#endif

DEPLOY_CPU(LpNormalizeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(LpNormalizeGradient);
#endif

OPERATOR_SCHEMA(LpNormalize)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(LpNormalizeGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(LpNormalize, GenericGradientMaker);

#undef CANONICALIZE_AXES_WITH_TENSOR

} // namespace dragon
