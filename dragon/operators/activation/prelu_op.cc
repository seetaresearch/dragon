#include "dragon/operators/activation/prelu_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void PReluOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  int64_t N, C, S;
  if (W.count() > 1) { // Channel-wise
    if (X.ndim() == 1) { // Per-activation
      N = S = 1, C = X.count();
    } else { // Spatial
      N = X.dim(0);
      if (data_format() == "NCHW") {
        C = X.dim(1), S = X.count(2);
      } else {
        C = X.dim(-1);
        S = X.count(1) / C;
      }
    }
  } else { // Channel-shared
    C = S = 1, N = X.count();
  }
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({C}), T);

  kernels::PRelu(
      N,
      S,
      C,
      S == 1 ? "NHWC" : data_format(),
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void PReluGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1);
  int64_t N, C, S;
  if (W.count() > 1) {
    if (X.ndim() == 1) {
      N = S = 1, C = X.count();
    } else {
      N = X.dim(0);
      if (data_format() == "NCHW") {
        C = X.dim(1), S = X.count(2);
      } else {
        C = X.dim(-1);
        S = X.count(1) / C;
      }
    }
  } else {
    C = S = 1, N = X.count();
  }

  if (dW->has_name()) {
    kernels::PReluWGrad(
        N,
        S,
        C,
        S == 1 ? "NHWC" : data_format(),
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dW->ReshapeLike(W)->template mutable_data<T, Context>(),
        ctx());
  }

  if (dX->has_name()) {
    kernels::PReluGrad(
        N,
        S,
        C,
        S == 1 ? "NHWC" : data_format(),
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        W.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(PRelu);
DEPLOY_CPU_OPERATOR(PReluGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(PRelu);
DEPLOY_CUDA_OPERATOR(PReluGradient);
#endif

OPERATOR_SCHEMA(PRelu)
    /* X, W */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(PReluGradient)
    /* X, W, dY */
    .NumInputs(3)
    /* dX, dW */
    .NumOutputs(2);

REGISTER_GRADIENT(PRelu, GenericGradientMaker);

} // namespace dragon
