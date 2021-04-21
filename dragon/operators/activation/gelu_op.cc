#include "dragon/operators/activation/gelu_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void GeluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  if (approximate_) {
    kernels::ApproxGelu(
        X.count(),
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    kernels::Gelu(
        X.count(),
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void GeluGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  if (approximate_) {
    kernels::ApproxGeluGrad(
        X.count(),
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    kernels::GeluGrad(
        X.count(),
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Gelu);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Gelu);
#endif

DEPLOY_CPU_OPERATOR(GeluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GeluGradient);
#endif

OPERATOR_SCHEMA(Gelu)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GeluGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Gelu, GenericGradientMaker);

} // namespace dragon
