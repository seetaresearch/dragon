#include "dragon/operators/normalization/lp_normalize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void LpNormalizeOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);
  auto reduce_dim = X.count(axis, end_axis + 1);

  // Normalize input with a scaled Lp-norm
  if (p_ == 1) {
    kernels::L1Normalize(
        X.count(0, axis),
        X.count(end_axis + 1),
        reduce_dim,
        reduction_ == "MEAN" ? float(reduce_dim) : 1.f,
        epsilon_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else if (p_ == 2) {
    kernels::L2Normalize(
        X.count(0, axis),
        X.count(end_axis + 1),
        reduce_dim,
        reduction_ == "MEAN" ? float(reduce_dim) : 1.f,
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
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void LpNormalizeGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);
  auto reduce_dim = X.count(axis, end_axis + 1);

  if (p_ == 1) {
    kernels::L1NormalizeGrad(
        X.count(0, axis),
        X.count(end_axis + 1),
        reduce_dim,
        reduction_ == "MEAN" ? float(reduce_dim) : 1.f,
        epsilon_,
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else if (p_ == 2) {
    kernels::L2NormalizeGrad(
        X.count(0, axis),
        X.count(end_axis + 1),
        reduce_dim,
        reduction_ == "MEAN" ? float(reduce_dim) : 1.f,
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
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(LpNormalize);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LpNormalize);
#endif

DEPLOY_CPU_OPERATOR(LpNormalizeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LpNormalizeGradient);
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

} // namespace dragon
