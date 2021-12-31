#include "dragon/operators/normalization/lp_norm_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void LpNormOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);
  auto reduce_dim = X.count(axis, end_axis + 1);

  if (p_ == 1) {
    kernels::L1Norm(
        X.count(0, axis),
        X.count(end_axis + 1),
        reduce_dim,
        reduction_ == "MEAN" ? float(reduce_dim) : 1.f,
        epsilon_,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else if (p_ == 2) {
    kernels::L2Norm(
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
template <typename T>
void LpNormGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);
  auto reduce_dim = X.count(axis, end_axis + 1);

  if (p_ == 1) {
    kernels::L1NormGrad(
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
    kernels::L2NormGrad(
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

DEPLOY_CPU_OPERATOR(LpNorm);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LpNorm);
#endif

DEPLOY_CPU_OPERATOR(LpNormGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LpNormGradient);
#endif

OPERATOR_SCHEMA(LpNorm)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(LpNormGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(LpNorm, GenericGradientMaker);

} // namespace dragon
